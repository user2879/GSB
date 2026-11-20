"""业务错误码与失败记录。所有错误码均为稳定字符串，供账本与诊断报告核对。"""

from __future__ import annotations

from dataclasses import dataclass

E_IMAGE_FORMAT = "E_IMAGE_FORMAT"
E_TRUTH_INVALID = "E_TRUTH_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_NONFINITE_FEATURE = "E_NONFINITE_FEATURE"
E_QUANT_LEVEL = "E_QUANT_LEVEL"
E_IO = "E_IO"

ALL_CODES = (
    E_IMAGE_FORMAT,
    E_TRUTH_INVALID,
    E_CONFIG_INVALID,
    E_POINT_BUDGET_PENDING,
    E_NONFINITE_FEATURE,
    E_QUANT_LEVEL,
    E_IO,
)


@dataclass(frozen=True)
class Failure:
    """单条失败记录。排序键保证失败清单跨运行完全一致。"""

    image: str
    stage: str
    code: str
    detail: str
    injection_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "stage": self.stage,
            "code": self.code,
            "detail": self.detail,
            "injection_id": self.injection_id,
        }

    @staticmethod
    def from_dict(data: dict) -> "Failure":
        return Failure(
            image=data["image"],
            stage=data["stage"],
            code=data["code"],
            detail=data["detail"],
            injection_id=data.get("injection_id"),
        )

    @staticmethod
    def sort_key(failure: "Failure") -> tuple:
        return (failure.image, failure.stage, failure.code, failure.detail,
                failure.injection_id or "")
