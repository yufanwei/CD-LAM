#!/usr/bin/env python3
"""Download official AgiBot Alpha samples or Apple EgoDex archives safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


AGIBOT_REPO = "agibot-world/AgiBotWorld-Alpha"
AGIBOT_PAGE = f"https://huggingface.co/datasets/{AGIBOT_REPO}"
AGIBOT_SOURCE = "https://github.com/OpenDriveLab/Agibot-World"
AGIBOT_SAMPLE = "sample_dataset.tar"
AGIBOT_REVISION = "128665c9e0244c45d1cbe5c13f5a4706afd24f27"
AGIBOT_SAMPLE_BYTES = 7_097_989_120
AGIBOT_SAMPLE_SHA256 = (
    "131c6f99ebe6900e93d56be9f0cbe46f2cff286b8d9102b8d3e01d25f7cebe5e"
)
AGIBOT_SOURCE_RECORD = "agibot_download.json"

EGODEX_SOURCE = "https://github.com/apple/ml-egodex"
EGODEX_BASE = "https://ml-site.cdn-apple.com/datasets/egodex"
EGODEX_PARTS = {
    "part1": (f"{EGODEX_BASE}/part1.zip", "about 300 GB"),
    "part2": (f"{EGODEX_BASE}/part2.zip", "about 300 GB"),
    "part3": (f"{EGODEX_BASE}/part3.zip", "about 300 GB"),
    "part4": (f"{EGODEX_BASE}/part4.zip", "about 300 GB"),
    "part5": (f"{EGODEX_BASE}/part5.zip", "about 300 GB"),
    "test": (f"{EGODEX_BASE}/test.zip", "about 16 GB"),
    "extra": (f"{EGODEX_BASE}/extra.zip", "about 200 GB"),
}


class DownloadError(RuntimeError):
    """Raised when an official download cannot be completed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"cannot read AgiBot source record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DownloadError("AgiBot source record must contain a JSON object")
    return value


def _verify_local_agibot_archive(
    archive: Path, record_path: Path, revision: str
) -> str:
    record = _read_source_record(record_path)
    if (
        record.get("dataset") != AGIBOT_REPO
        or record.get("filename") != AGIBOT_SAMPLE
        or record.get("revision") != revision
        or record.get("revision_verified") is not True
        or record.get("archive_bytes") != AGIBOT_SAMPLE_BYTES
        or record.get("archive_sha256") != AGIBOT_SAMPLE_SHA256
    ):
        raise DownloadError(
            "local AgiBot archive source record does not match the pinned dataset revision"
        )
    expected = record.get("archive_sha256")
    observed = _sha256(archive)
    if (
        archive.stat().st_size != AGIBOT_SAMPLE_BYTES
        or not isinstance(expected, str)
        or expected != observed
    ):
        raise DownloadError(
            "local AgiBot archive SHA-256 does not match its source record"
        )
    return observed


def _write_agibot_source_record(
    output: Path, archive: Path, revision: str, archive_sha256: str
) -> Path:
    try:
        archive_path = archive.resolve().relative_to(output.resolve()).as_posix()
    except ValueError:
        archive_path = str(archive.resolve())
    payload = {
        "schema_version": 1,
        "dataset": AGIBOT_REPO,
        "filename": AGIBOT_SAMPLE,
        "revision": revision,
        "revision_verified": True,
        "archive_path": archive_path,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
    }
    path = output / AGIBOT_SOURCE_RECORD
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def plans() -> dict[str, object]:
    """Return the public source and download inventory."""

    return {
        "agibot_alpha": {
            "official_source": AGIBOT_SOURCE,
            "official_dataset": AGIBOT_PAGE,
            "access": "gated; accept the dataset terms on Hugging Face first",
            "license": "CC BY-NC-SA 4.0 plus the gated community agreement",
            "sample": AGIBOT_SAMPLE,
            "revision": AGIBOT_REVISION,
            "sample_bytes": AGIBOT_SAMPLE_BYTES,
            "sample_sha256": AGIBOT_SAMPLE_SHA256,
            "sample_size": "about 7.1 GB",
            "full_size": (
                "about 8.5 TB of dataset content; the HF repository footprint "
                "may be larger"
            ),
        },
        "egodex": {
            "official_source": EGODEX_SOURCE,
            "access": "public Apple CDN",
            "dataset_license": "CC BY-NC-ND",
            "archives": {
                name: {"url": url, "size": size}
                for name, (url, size) in EGODEX_PARTS.items()
            },
        },
    }


def _progress(downloaded: int, expected: int | None) -> None:
    gib = downloaded / 1024**3
    if expected:
        print(f"downloaded {gib:.2f}/{expected / 1024**3:.2f} GiB", flush=True)
    else:
        print(f"downloaded {gib:.2f} GiB", flush=True)


def resumable_download(url: str, destination: Path) -> Path:
    """Download to ``.part`` with HTTP Range resume and an atomic rename."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "CD-LAM-dataset-downloader/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"download failed with HTTP {exc.code}: {url}") from exc
    status = getattr(response, "status", None)
    if offset and status != 206:
        response.close()
        offset = 0
        request = urllib.request.Request(
            url, headers={"User-Agent": headers["User-Agent"]}
        )
        response = urllib.request.urlopen(request, timeout=120)
    content_length = response.headers.get("Content-Length")
    expected = offset + int(content_length) if content_length else None
    mode = "ab" if offset else "wb"
    downloaded = offset
    next_report = downloaded + 1024**3
    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                _progress(downloaded, expected)
                next_report = downloaded + 1024**3
    if expected is not None and downloaded != expected:
        raise DownloadError(
            f"incomplete download: expected {expected} bytes, received {downloaded}"
        )
    partial.replace(destination)
    return destination


def _safe_extract_tar(archive: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with tarfile.open(archive) as handle:
        members = []
        for member in handle.getmembers():
            target = (output / member.name).resolve()
            if not target.is_relative_to(root):
                raise DownloadError(
                    f"tar member escapes output directory: {member.name}"
                )
            if member.issym() or member.islnk():
                raise DownloadError(f"tar links are not accepted: {member.name}")
            members.append(member)
        handle.extractall(output, members=members)


def download_agibot(args: argparse.Namespace) -> int:
    if not args.accept_license:
        raise DownloadError(
            "pass --accept-license after accepting the gated terms at " + AGIBOT_PAGE
        )
    if args.source_record is not None and args.local_archive is None:
        raise DownloadError("--source-record is valid only with --local-archive")
    plan = {
        "dataset": AGIBOT_REPO,
        "filename": AGIBOT_SAMPLE,
        "revision": args.revision,
        "output": str(args.output.resolve()),
        "extract": args.extract,
        "local_archive": str(args.local_archive.resolve())
        if args.local_archive
        else None,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if (
        args.revision != AGIBOT_REVISION
        or re.fullmatch(r"[0-9a-f]{40}", args.revision) is None
    ):
        raise DownloadError(
            f"this release pins AgiBot sample revision {AGIBOT_REVISION}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.local_archive:
        downloaded = args.local_archive.expanduser().resolve()
        if not downloaded.is_file():
            raise DownloadError(f"local AgiBot archive is missing: {downloaded}")
        if args.source_record is None:
            raise DownloadError("--local-archive requires its verified --source-record")
        archive_sha256 = _verify_local_agibot_archive(
            downloaded, args.source_record.expanduser().resolve(), args.revision
        )
    else:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        except ImportError as exc:
            raise DownloadError(
                "install the download extra first: python -m pip install -e '.[download]'"
            ) from exc
        try:
            downloaded = Path(
                hf_hub_download(
                    AGIBOT_REPO,
                    AGIBOT_SAMPLE,
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=args.output,
                )
            )
        except GatedRepoError as exc:
            raise DownloadError(
                "AgiBot Alpha access is gated. Accept the terms in a browser, then run "
                "`hf auth login` with that same account before retrying."
            ) from exc
        except HfHubHTTPError as exc:
            raise DownloadError(
                f"AgiBot download failed at pinned revision {args.revision}: {exc}"
            ) from exc
        archive_sha256 = _sha256(downloaded)
    if (
        downloaded.stat().st_size != AGIBOT_SAMPLE_BYTES
        or archive_sha256 != AGIBOT_SAMPLE_SHA256
    ):
        raise DownloadError(
            "AgiBot sample bytes do not match the release-pinned official LFS object"
        )
    record_path = _write_agibot_source_record(
        args.output, downloaded, args.revision, archive_sha256
    )
    print(downloaded)
    print(record_path)
    if args.extract:
        extracted = args.output / "sample_dataset"
        if extracted.exists():
            raise DownloadError(
                f"refusing to overwrite extraction directory: {extracted}"
            )
        _safe_extract_tar(downloaded, extracted)
        print(extracted)
    return 0


def download_egodex(args: argparse.Namespace) -> int:
    if not args.accept_license:
        raise DownloadError(
            "pass --accept-license after reviewing the EgoDex dataset terms at "
            + EGODEX_SOURCE
        )
    url, size = EGODEX_PARTS[args.part]
    archive = args.output / f"{args.part}.zip"
    plan = {
        "part": args.part,
        "url": url,
        "documented_size": size,
        "archive": str(archive.resolve()),
        "extraction_root": str((args.output / "extracted" / args.part).resolve()),
        "extract": args.extract,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    downloaded = resumable_download(url, archive)
    with zipfile.ZipFile(downloaded) as handle:
        broken = handle.testzip()
        if broken is not None:
            raise DownloadError(f"ZIP integrity check failed at member: {broken}")
        if args.extract:
            target = args.output / "extracted" / args.part
            if target.exists():
                raise DownloadError(
                    f"refusing to overwrite extraction directory: {target}"
                )
            target.mkdir(parents=True)
            root = target.resolve()
            for info in handle.infolist():
                path = (target / info.filename).resolve()
                if not path.is_relative_to(root):
                    raise DownloadError(
                        f"ZIP member escapes output directory: {info.filename}"
                    )
            handle.extractall(target)
            print(target)
    print(downloaded)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("links", help="print official sources, sizes, and licenses")

    agibot = subparsers.add_parser(
        "agibot-sample", help="download the gated 7.1 GB sample"
    )
    agibot.add_argument("--output", type=Path, required=True)
    agibot.add_argument("--revision", default=AGIBOT_REVISION)
    agibot.add_argument("--local-archive", type=Path)
    agibot.add_argument("--source-record", type=Path)
    agibot.add_argument("--accept-license", action="store_true")
    agibot.add_argument("--extract", action="store_true")
    agibot.add_argument("--dry-run", action="store_true")

    egodex = subparsers.add_parser("egodex", help="download one official Apple archive")
    egodex.add_argument("--part", choices=sorted(EGODEX_PARTS), required=True)
    egodex.add_argument("--output", type=Path, required=True)
    egodex.add_argument("--accept-license", action="store_true")
    egodex.add_argument("--extract", action="store_true")
    egodex.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "links":
        print(json.dumps(plans(), indent=2, sort_keys=True))
        return 0
    if args.command == "agibot-sample":
        return download_agibot(args)
    if args.command == "egodex":
        return download_egodex(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DownloadError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"download_datasets: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
