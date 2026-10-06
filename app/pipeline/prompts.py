from app.config import settings

# ------------------------------------------------------------------ PROMPTS
_SEMI_REAL = (
    "render the swapped head in the semi-realistic digital painting style of Picture 1: a realistic, true-to-life "
    "face with natural proportions and a recognisable likeness, painted with smooth airbrushed skin and soft "
    "realistic shading (not flat cartoon colors, not a photo and not a vector icon). subtle thin dark line work only "
    "around the eyes, lips, hair strands and jacket edges. detailed realistic eyes with crisp catchlights, soft "
    "glossy lips, natural eyebrows, fine individually painted hair strands with a glossy sheen, warm natural skin "
    "tones with gentle highlights, a strong red rim light along the hair, forehead and cheek edge, and cool blue "
    "fill light on the shadow side, matching the color palette and lighting of the jacket and emblem. the head "
    "blends seamlessly into the painting with no visible swap seam and no harsh photographic texture, "
    "and keeps exactly the face of Picture 2: same face shape, jaw, hairline, facial hair and the real eye color of "
    "Picture 2 (do not turn the eyes blue). "
    "keep the natural age, forehead lines, smile lines, under-eye detail and any grey in the beard or hair of "
    "Picture 2; the airbrushed finish must not slim the face, soften the jaw or change any proportion. "
    "sharp detail, high quality, 4k."
)

_COMIC = (
    "render the swapped head fully in the illustration style of Picture 1, not as a photo: a premium digital "
    "comic-book vector portrait with bold clean ink outlines of consistent line weight, cel-shaded skin with smooth "
    "airbrushed gradients and crisp highlight and shadow shapes, rich saturated warm skin tones, individually inked "
    "hair strands with a glossy sheen, glossy lips and sharp eye catchlights, a strong red rim light along the hair, "
    "forehead, cheek and jaw edge and cool blue fill shadows on the opposite side, matching the line quality, color "
    "palette and lighting of the jacket and emblem. the head blends seamlessly into the illustration with no "
    "photographic texture and no swap seam, while still looking exactly like the person in "
    "Picture 2: same face shape, hairline, facial hair and the real eye color of Picture 2 (do not turn the eyes "
    "blue). sharp detail, high quality, 4k."
)

STYLE_TEXT = (
    "the painted finish applies to the rendering only and must never change the face proportions. "
    + {"semi_real": _SEMI_REAL, "comic": _COMIC}.get(settings.STYLE_MODE, _COMIC)
    if settings.STYLE_MODE in ("semi_real", "comic")
    else "high quality, sharp details, 4k."
)

_EXPR = (
    "copy the head rotation and eye direction from Picture 1, but keep the facial expression and smile of Picture 2"
    if settings.KEEP_USER_EXPRESSION
    else "copy the direction of the eye, head rotation, micro expressions from Picture 1"
)

BFS = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
    "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly "
    "preserving the face, hair, eye color and nose structure of Picture 2. "
    + _EXPR
    + ". "
)

IDENTITY = (
    "IDENTITY IS THE TOP PRIORITY: the result must look like the same real person as Picture 2, not like the person "
    "or the face shape in Picture 1. copy the exact face geometry of Picture 2: overall face width-to-height ratio, "
    "forehead height and width, cheekbone position, full cheeks, jaw width, jawline angle and chin shape (never slim, "
    "narrow, sharpen, shorten or elongate the face, never make it more handsome, younger or more glamorous), the exact "
    "nose length, bridge and nostril width, the exact lip shape and thickness and the same mouth expression (a "
    "closed-mouth smile stays closed, an open smile with visible teeth stays open with the same visible teeth), the "
    "same eyebrow shape, thickness and spacing, the same eye shape, eye size and eye distance, the exact eye color of "
    "Picture 2 (dark brown stays dark brown, never lighten to blue, green or grey), the same skin tone, apparent age, "
    "smile lines and under-eye detail. keep every facial landmark in the same relative position "
    "as Picture 2; only the painting style comes from Picture 1. "
)

BUILD = (
    "BUILD: match the fullness of the person in Picture 2. if the face is round, full or heavy, keep it a little "
    "fuller: fuller cheeks, softer wider jaw, a fuller chin (even a soft double chin) and a thicker neck. if the face "
    "is thin or lean, keep it a little thinner: leaner cheeks, a defined narrower jaw, visible cheekbones and a "
    "slimmer neck. if average, keep it average. never slim down a full face and never fatten a thin face, and never "
    "copy the build of the person in Picture 1. "
)

FACE_CLEAN = (
    "FACE QUALITY: a clean, continuous, smooth jawline and chin contour from ear to chin on both sides, symmetrical and "
    "undistorted, joining the neck naturally, with no warped, melted, doubled, broken or jagged jaw outline. the skin "
    "is clean, smooth and evenly painted in one natural skin tone. "
)

VISOR_MANDATORY = (
    "CRITICAL REQUIREMENT: The face MUST wear the exact angular electric-blue wraparound visor "
    "from Picture 1 or 3. This is non-negotiable. No ordinary glasses, no sunglasses, no other eyewear. "
    "Only the futuristic visor with the clear transparent lens and blue frame. "
)

VISOR_LOOK_MAN = (
    "the visor is exactly the one worn in Picture 1: a single flat, angular, futuristic wraparound shield made of ONE "
    "continuous transparent panel, with a perfectly straight horizontal top edge running just under the eyebrows from "
    "the outer eye corner on one side, across the bridge of the nose, to the temple on the other side; sharp "
    "chamfered (cut, faceted) lower corners and a shallow notch for the nose; a thin light electric-blue frame line "
    "along its edges; a chunky faceted cyan-blue corner block with a white highlight at the outer end on the side "
    "nearest the camera; and a thick silver-white and blue arm running back along the temple to the ear on the far "
    "side. the lens is clear and transparent with only a very light icy-blue tint and a few crisp white diagonal "
    "glints, so the eyes, eyelashes, eyelids and eyebrows of Picture 2 stay sharp and fully visible through it. "
)

VISOR_LOOK_DEFAULT = (
    "the visor is exactly the large angular futuristic shield worn in Picture 1: ONE continuous transparent panel "
    "covering both eyes, with a straight thin electric-blue upper rim just under the eyebrows, broad faceted outer "
    "corners, a shallow V-shaped lower edge around the nose, cyan-blue side blocks and slim arms returning to both "
    "temples. the lens is clear and transparent with only a very light icy-blue tint and a few crisp white glints, "
    "so the eyes, eyelashes, eyelids and eyebrows of Picture 2 stay sharp and fully visible through it. do not turn "
    "it into ordinary eyeglasses or two separate lenses. "
)

VISOR_REF_TEXT = (
    "Picture 3 is a close-up of the exact visor glasses: copy this visor exactly onto the face (same shape, straight "
    "top edge, angular corners, blue frame, clear lens, corner blocks and side arms). use ONLY the glasses from "
    "Picture 3, never its face. "
)

VISOR_CORE = (
    "EYEWEAR (mandatory): the face wears the Picture 1 visor and NOTHING else. if Picture 2 wears eyeglasses of any "
    "kind (black, thick, round, rectangular, thin metal or clear), DELETE them: erase their frames, rims, nose pads, "
    "arms and lens reflections completely and paint bare natural skin where they were, then put the visor on top. "
)

VISOR_NEG = (
    "exactly ONE pair of glasses on the face: no black frame, no thick rims, no second frame, no double lines above "
    "or below the visor, no round or rectangular eyeglasses, no rounded goggles, no sunglasses, no opaque or solid "
    "blue lens. "
)

VISOR_FIT = (
    "fit the visor to the face of Picture 2: its width spans exactly from temple to temple of that face and is never "
    "wider than the face, it sits level across the bridge of the nose directly over both eyes with the eyes centered "
    "behind the lens, it follows the same head tilt and perspective as the face, and its arms end at the temples and "
    "ears. not oversized, not floating, not tilted, not sliding off the face. "
)

VISOR_RETRY = (
    "REMINDER: ordinary or thick eyeglasses must NOT appear anywhere; only the angular blue-framed clear wraparound "
    "visor of Picture 1 appears on the face, clearly visible. "
)

HIJAB_VISOR_NOTE = (
    "the visor sits on the face inside the hijab opening: it is no wider than the visible face between the two edges "
    "of the hijab, its arms and corner blocks tuck against the hijab at the temples (they may rest slightly over the "
    "fabric edge), the hijab edge above the eyebrows stays fully visible and is not covered, and the visor never "
    "sticks out beyond the hijab outline or covers the forehead fabric. the hijab fabric is not deformed by it. the "
    "visor MUST be clearly visible on her face. "
)

HIJAB_USER = (
    "she wears her own hijab from Picture 2, copied exactly: the same fabric colors (including any two-tone or "
    "lighter under-scarf), the same sheen and soft folds, wrapped the same way tightly around the face with the "
    "edge sitting at the same place on the forehead, cheeks and under the chin, covering all hair, both ears and the "
    "neck, then falling in soft drapes onto the shoulders and chest and flowing into the jacket collar of Picture 1. "
    "no hair visible, no pins, brooches or patterns added, the fabric is not changed to another color. only her head "
    "and hijab are taken from Picture 2; ignore her clothes, cardigan, shirt, body and background. "
)

HIJAB_FACE = (
    "keep her natural look from Picture 2: the full natural face with its full cheeks and soft jaw (do not slim or "
    "shrink it), her natural skin tone and smile lines, her natural makeup level and "
    "lip color only - do NOT add eyeliner, eyeshadow, long lashes, contouring or glamour retouching, and do not make "
    "her look younger. the hijab frames the face exactly like Picture 2, not looser and not further back. "
)

FEMALE_FACE = (
    "FEMALE FACE DETAILS: copy her face details exactly from Picture 2: the eyebrow shape, thickness and arch, the eye "
    "shape, size and eyelid fold, her natural lash level, the nose width, bridge and tip, the lip shape, fullness and "
    "natural lip color, the smile with the same visible teeth, the cheek fullness, the forehead, the face shape and "
    "chin, her skin tone and dimples. use only a subtle natural makeup level like Picture 2; do not "
    "add glamour makeup, strong lipstick, long lashes or contouring, and do not make her look younger or thinner. "
    "from Picture 1 take only the hairstyle, jacket, painting style and visor. "
)

HAIR_USER = (
    "HAIR: ignore the hairstyle of Picture 1 completely and keep the exact hair of Picture 2: the same color (never "
    "add red, orange or blond highlights), length, texture (curly, wavy or straight), volume, hairline, parting and "
    "side length. if Picture 2 has short hair keep it short; if Picture 2 is bald, shaved or has a receding hairline "
    "keep the scalp bald or receding exactly as in Picture 2 and do NOT add, grow or paint any hair. do not smooth, "
    "slick back, comb up, restyle or thicken the hair. the only change allowed on the hair is a thin red rim light. "
)

def visor_block(info, attempt=0, ref=False):
    look = VISOR_LOOK_MAN if info.get("avatar") == "Man" else VISOR_LOOK_DEFAULT
    return (
        VISOR_CORE
        + (VISOR_REF_TEXT if ref else "")
        + look
        + VISOR_NEG
        + VISOR_FIT
        + (VISOR_RETRY if attempt > 0 else "")
    )

def build_prompt(info, attempt=0, ref=False, style_mode=None, keep_user_expression=None):
    if style_mode is None:
        style_mode = settings.STYLE_MODE
    if keep_user_expression is None:
        keep_user_expression = settings.KEEP_USER_EXPRESSION

    expr = (
        "copy the head rotation and eye direction from Picture 1, but keep the facial expression and smile of Picture 2"
        if keep_user_expression
        else "copy the direction of the eye, head rotation, micro expressions from Picture 1"
    )
    bfs = (
        "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly "
        "preserving the face, hair, eye color and nose structure of Picture 2. "
        + expr
        + ". "
    )
    style_text = (
        (
            "the painted finish applies to the rendering only and must never change the face proportions. "
            + {"semi_real": _SEMI_REAL, "comic": _COMIC}[style_mode]
        )
        if style_mode in ("semi_real", "comic")
        else "high quality, sharp details, 4k."
    )
    V = visor_block(info, attempt, ref)
    closing = (
        "keep the jacket, emblem and solid black background of Picture 1 unchanged. "
        "final check: the face and build match Picture 2, the jawline is clean, and the angular blue clear "
        "visor of the avatar is on the face with no other glasses."
    )
    if info.get("hijab", False):
        return (
            f"{VISOR_MANDATORY}{bfs}{IDENTITY}{BUILD}{V}{HIJAB_VISOR_NOTE}"
            f"{HIJAB_FACE}{HIJAB_USER}{FACE_CLEAN}{style_text} {closing} the visor is present and fits her face."
        )
    if info.get("gender") == "Woman":
        return (
            f"{VISOR_MANDATORY}{bfs}{IDENTITY}{FEMALE_FACE}{BUILD}"
            "her hair is styled exactly like Picture 1: long black hair with red and orange highlights woven "
            "throughout, styled in a high voluminous bun or updo at the crown, sleek and professionally polished, "
            f"framing the face. {V}{FACE_CLEAN}{style_text} " + closing
        )
    return (
        f"{VISOR_MANDATORY}{bfs}{IDENTITY}{BUILD}{HAIR_USER}"
        "FACIAL HAIR: keep the facial hair of Picture 2 exactly as it is: if Picture 2 has a moustache, goatee, "
        "beard or stubble keep the same shape, coverage, length, density and grey or black color with a neat "
        "natural edge; if Picture 2 is clean-shaven keep the skin smooth and add no facial hair. do not copy any "
        f"facial hair from Picture 1. {V}{FACE_CLEAN}{style_text} " + closing
    )
