from __future__ import annotations

import argparse
from pathlib import Path

DEMO_TEXT = """第一回 甄士隐梦幻识通灵

甄士隐住在姑苏城中，家境殷实，性情恬淡。
一日炎夏永昼，他在书房中倦睡，梦中遇见一僧一道。

第二回 贾夫人仙逝扬州城

林如海之女名黛玉，自幼聪慧，体弱多病。
贾雨村因缘际会，后来得以复起为官。

第三回 贾雨村夤缘复旧职

黛玉进京，初入荣国府，拜见外祖母贾母。
宝玉闻得来了一个神仙似的妹妹，心中十分欢喜。
"""


def count_paragraphs(text: str) -> int:
    return sum(1 for part in text.splitlines() if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny demo corpus for Phase 1.")
    parser.add_argument(
        "--out",
        default="rag_tutor/data/demo_hongloumeng.txt",
        help="Output path for the demo text.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(DEMO_TEXT, encoding="utf-8")

    chapters = [line for line in DEMO_TEXT.splitlines() if line.startswith(("第一回", "第二回", "第三回"))]
    print(f"demo_path: {out_path}")
    print(f"chars: {len(DEMO_TEXT)}")
    print(f"non_empty_paragraphs: {count_paragraphs(DEMO_TEXT)}")
    print(f"chapters: {len(chapters)}")


if __name__ == "__main__":
    main()

