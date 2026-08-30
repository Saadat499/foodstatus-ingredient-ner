def segment_ingredient_list(text):
    current = []
    results = []
    segments = []
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(char)

    if len(current) > 0:
        segments.append("".join(current))


    for segment in segments:
        parsed = parse_segment(segment)
        results.append(parsed)

    return results


def flag_hard_cases(segments: list[dict]) -> list[dict]:
    for seg in segments:
        raw_lower = seg["raw"].lower()
        seg["needs_ner_review"] = (
                    "may contain" in raw_lower
                    or "traces of" in raw_lower
                    or (seg["qualifier"] and "," in seg["qualifier"])
                    or len(seg["name"].split()) >= 4
        )

    return segments





def parse_segment(segment):
    seg = segment.strip()
    bracket_index = seg.find("(")
    closed_bracket_index = seg.find(")")


    if bracket_index == -1:
        name = seg
        qualifier = None       
    else:
        name_bracket_index = seg[:bracket_index]
        name = name_bracket_index.strip()
        qualifier = seg[bracket_index + 1 : closed_bracket_index]

    return {"raw" : segment, "name" : name, "qualifier" : qualifier}

if __name__ == "__main__":
    test_input = (
        "Water, Sugar, Gelatin (Bovine), E471, Natural Flavor (Contains Soy), "
        "Modified Corn Starch, May contain traces of tree nuts, "
        "Enzyme Treated Starch (Microbial Source)"
    )
    parsed = flag_hard_cases(segment_ingredient_list(test_input))
    for p in parsed:
        print(p)


    

