import json

from PIL import Image

from glance_retrieval.audit import audit_manifest


def test_audit_detects_duplicate_images(tmp_path):
    image_path = tmp_path / "a.png"
    duplicate_path = tmp_path / "b.png"
    Image.new("RGB", (32, 24), (240, 20, 20)).save(image_path)
    Image.new("RGB", (32, 24), (240, 20, 20)).save(duplicate_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"image_id": "a", "path": str(image_path), "metadata": {}}),
                json.dumps({"image_id": "b", "path": str(duplicate_path), "metadata": {}}),
            ]
        ),
        encoding="utf-8",
    )
    report = audit_manifest(manifest, expected_minimum=2)
    assert report["valid_images"] == 2
    assert report["duplicate_groups"] == [["a", "b"]]
    assert not report["passed"]
