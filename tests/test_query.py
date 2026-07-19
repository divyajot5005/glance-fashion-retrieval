from glance_retrieval.query import decompose_query


def test_compositional_query_keeps_attribute_bindings():
    plan = decompose_query("A red tie and a white shirt in a formal office")
    assert plan.atomic_attributes == ("red tie", "white shirt")
    assert plan.contexts == ("office",)
    assert plan.styles == ("formal",)
    assert plan.is_compositional
    assert plan.fashion_weight > plan.context_weight


def test_context_dominates_for_place_only_query():
    plan = decompose_query("people walking outdoors in a park")
    assert plan.context_weight > plan.fashion_weight


def test_unknown_language_falls_back_to_full_query():
    query = "an avant-garde monochromatic silhouette"
    plan = decompose_query(query)
    assert plan.fashion_text == query
    assert plan.context_text == query

