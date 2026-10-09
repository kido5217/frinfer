from __future__ import annotations

from tools.bench.ttft import cases
from tools.bench.ttft.corpus import Corpus


def test_checked_in_manifest_constructs() -> None:
    """The real Corpus must validate against the committed manifest.

    The other Serve TTFT tests substitute a local Corpus stub, so a manifest
    that drops a shape required by ``corpus.validate()`` (or a shape referenced
    by a registered case) is otherwise only observed when the benchmark runs.
    """
    Corpus()


def test_registered_case_corpus_ids_resolve() -> None:
    corpus = Corpus()
    shapes = corpus.manifest["shapes"]
    shared = corpus.manifest["shared"]
    media = corpus.manifest["media"]
    load_images = media["load-images"]

    for definition in cases.CASES.values():
        for corpus_id in definition.corpus_ids:
            if corpus_id.startswith("load-image-"):
                index = int(corpus_id.removeprefix("load-image-"))
                assert 0 <= index < len(load_images), (
                    f"{definition.name} references unknown {corpus_id!r}"
                )
                continue
            assert corpus_id in shapes or corpus_id in shared or corpus_id in media, (
                f"{definition.name} references unknown corpus fixture {corpus_id!r}"
            )
