import re
import json

# Configuration: symbol pools
ear_pairs = [("(", ")"), ("[", "]"), ("༼", "༽"), ("﴾", "﴿"), ("ᕦ", "ᕤ"), ("<", ">")]
cheek_pairs = [("ღ", "ღ"), ("╚", "╚"), ("✿", "✿"), ("｡", "｡"), ("✧", "✧"), ("ミ", "ミ")]
eyes = ["◉", "ⱺ", "❂", "♥", "▓", "ㅅ", "⚙", "▰", "◕", "⍜", "⨱"]
noses = ["V", "v", "-", "ᴥ", "▾", "ᚖ", "﹏", "ㅅ", "一", "ヮ", "益", "口"]

def parse_glyph(glyph: str):
    # Find ear pair used
    for el, er in ear_pairs:
        pattern = re.escape(el) + r"{2}"
        match = re.search(pattern, glyph)
        if match:
            i = match.start()
            # header is everything before el*2
            header = glyph[:i]
            # skip el*2
            rest = glyph[i + len(el)*2:]
            # find footer by searching er*2 at end
            er_pattern = re.escape(er)*2
            m2 = re.search(er_pattern, rest)
            if m2:
                mid = rest[:m2.start()]
                footer = rest[m2.start()+len(er)*2:]
                # inside mid: cheek left (first char), eye, nose, eye, cheek right (last char)
                cheek_left = mid[0]
                cheek_right = mid[-1]
                eye1 = mid[1]
                nose = mid[2]
                eye2 = mid[3]
                return {
                    "glyph": glyph,
                    "header": header,
                    "aside_left": el,
                    "widget_left": cheek_left,
                    "content_main": eye1 + nose + eye2,
                    "widget_right": cheek_right,
                    "aside_right": er,
                    "footer": footer
                }
    # fallback: return raw
    return {"glyph": glyph}

# Example usage with provided glyphs
examples = [
    "ʕᵒ☯ϖ☯ᵒʔ", "乁﴾ovo乁﴿", "୧ˇ☯-☯ˇ୨", "ʢᵒᴗ.ᴗᵒʡ", "༼⌐■∇■¬༽",
    "ᕳ>人<ᕲ", "ᕕ|✧ڡ✧ᕕ|", "ᖗᵒ͌͜ʖ͌ᵒᖘζ༼ᵔϖᔊζ༽"
]

# Generate JSONL
for glyph in examples:
    parsed = parse_glyph(glyph)
    print(json.dumps(parsed, ensure_ascii=False))
