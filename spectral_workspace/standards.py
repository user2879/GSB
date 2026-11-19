"""Vendored CIE observer tables and deterministic colour constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .canonical import sha256_obj
from .errors import DomainError, E_OBSERVER_INVALID


WAVELENGTHS_NM: tuple[int, ...] = tuple(range(380, 781, 5))
INTEGRATION_STEP_NM = 5
SAMPLE_COUNT = 81
CMF_TABLE_VERSION = "cie-standard-observer-5nm-380-780-v1"
WAVELENGTH_GRID_HASH = sha256_obj(list(WAVELENGTHS_NM))


def _q(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


# Data source: CIE 1931 2 degree and CIE 1964 10 degree standard observers,
# sampled at 5 nm and restricted to the application's 380-780 nm domain.
# Values are vendored so runtime behaviour never depends on network access.
CMF_DATA: dict[str, tuple[tuple[float, float, float], ...]] = {
    "CIE1931_2": (
        (0.001368000, 0.000039000, 0.006450001),
        (0.002236000, 0.000064000, 0.010549990),
        (0.004243000, 0.000120000, 0.020050010),
        (0.007650000, 0.000217000, 0.036210000),
        (0.014310000, 0.000396000, 0.067850010),
        (0.023190000, 0.000640000, 0.110200000),
        (0.043510000, 0.001210000, 0.207400000),
        (0.077630000, 0.002180000, 0.371300000),
        (0.134380000, 0.004000000, 0.645600000),
        (0.214770000, 0.007300000, 1.039050100),
        (0.283900000, 0.011600000, 1.385600000),
        (0.328500000, 0.016840000, 1.622960000),
        (0.348280000, 0.023000000, 1.747060000),
        (0.348060000, 0.029800000, 1.782600000),
        (0.336200000, 0.038000000, 1.772110000),
        (0.318700000, 0.048000000, 1.744100000),
        (0.290800000, 0.060000000, 1.669200000),
        (0.251100000, 0.073900000, 1.528100000),
        (0.195360000, 0.090980000, 1.287640000),
        (0.142100000, 0.112600000, 1.041900000),
        (0.095640000, 0.139020000, 0.812950100),
        (0.057950010, 0.169300000, 0.616200000),
        (0.032010000, 0.208020000, 0.465180000),
        (0.014700000, 0.258600000, 0.353300000),
        (0.004900000, 0.323000000, 0.272000000),
        (0.002400000, 0.407300000, 0.212300000),
        (0.009300000, 0.503000000, 0.158200000),
        (0.029100000, 0.608200000, 0.111700000),
        (0.063270000, 0.710000000, 0.078249990),
        (0.109600000, 0.793200000, 0.057250010),
        (0.165500000, 0.862000000, 0.042160000),
        (0.225749900, 0.914850100, 0.029840000),
        (0.290400000, 0.954000000, 0.020300000),
        (0.359700000, 0.980300000, 0.013400000),
        (0.433449900, 0.994950100, 0.008749999),
        (0.512050100, 1.000000000, 0.005749999),
        (0.594500000, 0.995000000, 0.003900000),
        (0.678400000, 0.978600000, 0.002749999),
        (0.762100000, 0.952000000, 0.002100000),
        (0.842500000, 0.915400000, 0.001800000),
        (0.916300000, 0.870000000, 0.001650001),
        (0.978600000, 0.816300000, 0.001400000),
        (1.026300000, 0.757000000, 0.001100000),
        (1.056700000, 0.694900000, 0.001000000),
        (1.062200000, 0.631000000, 0.000800000),
        (1.045600000, 0.566800000, 0.000600000),
        (1.002600000, 0.503000000, 0.000340000),
        (0.938400000, 0.441200000, 0.000240000),
        (0.854449900, 0.381000000, 0.000190000),
        (0.751400000, 0.321000000, 0.000100000),
        (0.642400000, 0.265000000, 0.000050000),
        (0.541900000, 0.217000000, 0.000030000),
        (0.447900000, 0.175000000, 0.000020000),
        (0.360800000, 0.138200000, 0.000010000),
        (0.283500000, 0.107000000, 0.000000000),
        (0.218700000, 0.081600000, 0.000000000),
        (0.164900000, 0.061000000, 0.000000000),
        (0.121200000, 0.044580000, 0.000000000),
        (0.087400000, 0.032000000, 0.000000000),
        (0.063600000, 0.023200000, 0.000000000),
        (0.046770000, 0.017000000, 0.000000000),
        (0.032900000, 0.011920000, 0.000000000),
        (0.022700000, 0.008210000, 0.000000000),
        (0.015840000, 0.005723000, 0.000000000),
        (0.011359160, 0.004102000, 0.000000000),
        (0.008110916, 0.002929000, 0.000000000),
        (0.005790346, 0.002091000, 0.000000000),
        (0.004109457, 0.001484000, 0.000000000),
        (0.002899327, 0.001047000, 0.000000000),
        (0.002049190, 0.000740000, 0.000000000),
        (0.001439971, 0.000520000, 0.000000000),
        (0.000999949, 0.000361100, 0.000000000),
        (0.000690079, 0.000249200, 0.000000000),
        (0.000476021, 0.000171900, 0.000000000),
        (0.000332301, 0.000120000, 0.000000000),
        (0.000234826, 0.000084800, 0.000000000),
        (0.000166151, 0.000060000, 0.000000000),
        (0.000117413, 0.000042400, 0.000000000),
        (0.000083075, 0.000030000, 0.000000000),
        (0.000058707, 0.000021200, 0.000000000),
        (0.000041510, 0.000014990, 0.000000000),
    ),
    "CIE1964_10": (
        (0.000159952, 0.000017364, 0.000704776),
        (0.000662440, 0.000071560, 0.002927800),
        (0.002361600, 0.000253400, 0.010482200),
        (0.007242300, 0.000768500, 0.032344000),
        (0.019109700, 0.002004400, 0.086010900),
        (0.043400000, 0.004509000, 0.197120000),
        (0.084736000, 0.008756000, 0.389366000),
        (0.140638000, 0.014456000, 0.656760000),
        (0.204492000, 0.021391000, 0.972542000),
        (0.264737000, 0.029497000, 1.282500000),
        (0.314679000, 0.038676000, 1.553480000),
        (0.357719000, 0.049602000, 1.798500000),
        (0.383734000, 0.062077000, 1.967280000),
        (0.386726000, 0.074704000, 2.027300000),
        (0.370702000, 0.089456000, 1.994800000),
        (0.342957000, 0.106256000, 1.900700000),
        (0.302273000, 0.128201000, 1.745370000),
        (0.254085000, 0.152761000, 1.554900000),
        (0.195618000, 0.185190000, 1.317560000),
        (0.132349000, 0.219940000, 1.030200000),
        (0.080507000, 0.253589000, 0.772125000),
        (0.041072000, 0.297665000, 0.570060000),
        (0.016172000, 0.339133000, 0.415254000),
        (0.005132000, 0.395379000, 0.302356000),
        (0.003816000, 0.460777000, 0.218502000),
        (0.015444000, 0.531360000, 0.159249000),
        (0.037465000, 0.606741000, 0.112044000),
        (0.071358000, 0.685660000, 0.082248000),
        (0.117749000, 0.761757000, 0.060709000),
        (0.172953000, 0.823330000, 0.043050000),
        (0.236491000, 0.875211000, 0.030451000),
        (0.304213000, 0.923810000, 0.020584000),
        (0.376772000, 0.961988000, 0.013676000),
        (0.451584000, 0.982200000, 0.007918000),
        (0.529826000, 0.991761000, 0.003988000),
        (0.616053000, 0.999110000, 0.001091000),
        (0.705224000, 0.997340000, 0.000000000),
        (0.793832000, 0.982380000, 0.000000000),
        (0.878655000, 0.955552000, 0.000000000),
        (0.951162000, 0.915175000, 0.000000000),
        (1.014160000, 0.868934000, 0.000000000),
        (1.074300000, 0.825623000, 0.000000000),
        (1.118520000, 0.777405000, 0.000000000),
        (1.134300000, 0.720353000, 0.000000000),
        (1.123990000, 0.658341000, 0.000000000),
        (1.089100000, 0.593878000, 0.000000000),
        (1.030480000, 0.527963000, 0.000000000),
        (0.950740000, 0.461834000, 0.000000000),
        (0.856297000, 0.398057000, 0.000000000),
        (0.754930000, 0.339554000, 0.000000000),
        (0.647467000, 0.283493000, 0.000000000),
        (0.535110000, 0.228254000, 0.000000000),
        (0.431567000, 0.179828000, 0.000000000),
        (0.343690000, 0.140211000, 0.000000000),
        (0.268329000, 0.107633000, 0.000000000),
        (0.204300000, 0.081187000, 0.000000000),
        (0.152568000, 0.060281000, 0.000000000),
        (0.112210000, 0.044096000, 0.000000000),
        (0.081260600, 0.031800400, 0.000000000),
        (0.057930000, 0.022601700, 0.000000000),
        (0.040850800, 0.015905100, 0.000000000),
        (0.028623000, 0.011130300, 0.000000000),
        (0.019941300, 0.007748800, 0.000000000),
        (0.013842000, 0.005375100, 0.000000000),
        (0.009576880, 0.003717740, 0.000000000),
        (0.006605200, 0.002564560, 0.000000000),
        (0.004552630, 0.001768470, 0.000000000),
        (0.003144700, 0.001222390, 0.000000000),
        (0.002174960, 0.000846190, 0.000000000),
        (0.001505700, 0.000586440, 0.000000000),
        (0.001044760, 0.000407410, 0.000000000),
        (0.000727450, 0.000284041, 0.000000000),
        (0.000508258, 0.000198730, 0.000000000),
        (0.000356380, 0.000139550, 0.000000000),
        (0.000250969, 0.000098428, 0.000000000),
        (0.000177730, 0.000069819, 0.000000000),
        (0.000126390, 0.000049737, 0.000000000),
        (0.000090151, 0.000035541, 0.000000000),
        (0.000064526, 0.000025486, 0.000000000),
        (0.000046339, 0.000018338, 0.000000000),
        (0.000033412, 0.000013249, 0.000000000),
    ),
}


OBSERVER_DISPLAY_NAMES = {
    "CIE1931_2": "CIE 1931 2 Degree Standard Observer",
    "CIE1964_10": "CIE 1964 10 Degree Standard Observer",
}

OBSERVER_ALIASES = {
    "cie1931_2": "CIE1931_2",
    "cie1931_2deg": "CIE1931_2",
    "cie 1931 2 degree standard observer": "CIE1931_2",
    "cie1964_10": "CIE1964_10",
    "cie1964_10deg": "CIE1964_10",
    "cie 1964 10 degree standard observer": "CIE1964_10",
}


def normalize_observer(value: object, *, pointer: str | None = None) -> str:
    if not isinstance(value, str):
        raise DomainError(
            E_OBSERVER_INVALID,
            "observer must be a string",
            pointer=pointer,
            details={"raw": value},
        )
    if value in OBSERVER_DISPLAY_NAMES:
        return value
    normalized = OBSERVER_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise DomainError(
            E_OBSERVER_INVALID,
            "unsupported observer",
            pointer=pointer,
            details={
                "raw": value,
                "allowed": sorted(OBSERVER_DISPLAY_NAMES),
            },
        )
    return normalized


def cmf_table(observer: str) -> tuple[tuple[float, float, float], ...]:
    canonical = normalize_observer(observer)
    return CMF_DATA[canonical]


def cmf_hash(observer: str) -> str:
    canonical = normalize_observer(observer)
    return sha256_obj(
        {
            "version": CMF_TABLE_VERSION,
            "observer": canonical,
            "wavelengths_nm": list(WAVELENGTHS_NM),
            "values": [list(row) for row in CMF_DATA[canonical]],
        }
    )


# Standard illuminant white points normalised to Y=1.
D65_WHITE_XYZ: tuple[float, float, float] = (0.95047, 1.0, 1.08883)
D50_WHITE_XYZ: tuple[float, float, float] = (0.96422, 1.0, 0.82521)


@dataclass(frozen=True)
class AdaptationBasis:
    name: str
    matrix: tuple[tuple[float, float, float], ...]
    inverse: tuple[tuple[float, float, float], ...]


# The two bases are the standard Bradford and von Kries chromatic adaptation
# matrices.  Their inverses are embedded to avoid LAPACK-dependent last-bit
# differences between operating systems.
ADAPTATION_BASES: dict[str, AdaptationBasis] = {
    "bradford": AdaptationBasis(
        name="bradford",
        matrix=(
            (0.8951, 0.2664, -0.1614),
            (-0.7502, 1.7135, 0.0367),
            (0.0389, -0.0685, 1.0296),
        ),
        inverse=(
            (0.9869929054667123, -0.1470542564209901, 0.1599626516637312),
            (0.4323052697233945, 0.5183602715367774, 0.0492912282128556),
            (-0.0085286645751773, 0.0400428216540849, 0.9684866957875502),
        ),
    ),
    "von_kries": AdaptationBasis(
        name="von_kries",
        matrix=(
            (0.40024, 0.7076, -0.08081),
            (-0.2263, 1.16532, 0.0457),
            (0.0, 0.0, 0.91822),
        ),
        inverse=(
            (1.8599363874558397, -1.1293816185800916, 0.2198974095961933),
            (0.3611914362417676, 0.6388124632850422, -0.0000062561545857),
            (0.0, 0.0, 1.0890636230968613),
        ),
    ),
}


def normalize_adaptation_method(value: object, *, pointer: str | None = None) -> str:
    if not isinstance(value, str):
        raise DomainError(
            "E_ADAPTATION_INVALID",
            "adaptation method must be a string",
            pointer=pointer,
            details={"raw": value},
        )
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"none": "none", "identity": "none"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"bradford", "von_kries", "none"}:
        raise DomainError(
            "E_ADAPTATION_INVALID",
            "unsupported chromatic adaptation method",
            pointer=pointer,
            details={"raw": value, "allowed": ["bradford", "von_kries", "none"]},
        )
    return normalized


@dataclass(frozen=True)
class ColorSpace:
    name: str
    matrix_xyz_to_linear_rgb: tuple[tuple[float, float, float], ...] | None
    transfer: str
    white_xyz: tuple[float, float, float]


COLOR_SPACES: dict[str, ColorSpace] = {
    "XYZ": ColorSpace(
        name="XYZ", matrix_xyz_to_linear_rgb=None, transfer="linear", white_xyz=D65_WHITE_XYZ
    ),
    "sRGB": ColorSpace(
        name="sRGB",
        matrix_xyz_to_linear_rgb=(
            (3.2406, -1.5372, -0.4986),
            (-0.9689, 1.8758, 0.0415),
            (0.0557, -0.2040, 1.0570),
        ),
        transfer="sRGB IEC 61966-2-1",
        white_xyz=D65_WHITE_XYZ,
    ),
    "DisplayP3": ColorSpace(
        name="DisplayP3",
        matrix_xyz_to_linear_rgb=(
            (2.493496911941425, -0.9313836179191239, -0.40271078445071684),
            (-0.8294889695615747, 1.7626640603183463, 0.023624685841943577),
            (0.03584583024378447, -0.07617238926804182, 0.9568845240076872),
        ),
        transfer="IEC 61966-2-1 sRGB transfer (P3 primaries)",
        white_xyz=D65_WHITE_XYZ,
    ),
}

COLOR_SPACE_ALIASES = {
    "xyz": "XYZ",
    "srgb": "sRGB",
    "displayp3": "DisplayP3",
    "display_p3": "DisplayP3",
    "p3": "DisplayP3",
}


def normalize_color_space(value: object, *, pointer: str | None = None) -> str:
    if not isinstance(value, str):
        raise DomainError(
            "E_COLOR_SPACE_INVALID",
            "target colour space must be a string",
            pointer=pointer,
            details={"raw": value},
        )
    if value in COLOR_SPACES:
        return value
    normalized = COLOR_SPACE_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise DomainError(
            "E_COLOR_SPACE_INVALID",
            "unsupported target colour space",
            pointer=pointer,
            details={"raw": value, "allowed": sorted(COLOR_SPACES)},
        )
    return normalized


def mat3_mul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(float(left[row][k]) * float(right[k][col]) for k in range(3)) for col in range(3))
        for row in range(3)
    )


def mat3_vec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    )


def normalize_white_point(vector: Sequence[float]) -> tuple[float, float, float]:
    y = float(vector[1])
    return (float(vector[0]) / y, 1.0, float(vector[2]) / y)
