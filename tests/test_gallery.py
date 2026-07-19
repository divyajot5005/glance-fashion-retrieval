from glance_retrieval.gallery import write_gallery
from glance_retrieval.types import SearchResult


def test_gallery_is_self_contained_when_image_is_missing(tmp_path):
    output = tmp_path / "results.html"
    result = SearchResult(
        rank=1,
        image_id="demo",
        path=str(tmp_path / "missing.jpg"),
        score=0.9,
        fashion_global=0.8,
        fashion_atomic=0.9,
        context=0.7,
        metadata_bonus=0.0,
        metadata={},
    )
    write_gallery("red tie", [result], output)
    text = output.read_text(encoding="utf-8")
    assert "red tie" in text
    assert "Image unavailable" in text
