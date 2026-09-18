"""Capability-driven DirectShow controls. All COM calls belong to the camera worker."""
import math
import os


# Property identifiers from strmif.h; never guess ranges or auto-mode values.
PROPERTIES = {
    "exposure": ("camera", 4, "曝光", False),
    "gain": ("video", 9, "增益", False),
    "white_balance": ("video", 7, "白平衡", False),
    "focus": ("camera", 6, "對焦", False),
    "brightness": ("video", 0, "亮度", True),
    "contrast": ("video", 1, "對比", True),
    "saturation": ("video", 3, "飽和度", True),
    "sharpness": ("video", 4, "銳利度", True),
    "gamma": ("video", 5, "Gamma", True),
    "backlight": ("video", 8, "背光補償", True),
}


def validate_values(values):
    if not isinstance(values, dict) or len(values) > len(PROPERTIES):
        raise ValueError("相機參數必須為物件")
    for key, item in values.items():
        if key not in PROPERTIES or not isinstance(item, dict):
            raise ValueError("未知的相機參數")
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not -2147483648 <= value <= 2147483647 or not math.isfinite(value) or int(value) != value:
            raise ValueError("相機參數必須為有限整數")
        if not isinstance(item.get("auto"), bool):
            raise ValueError("相機自動模式必須為布林值")
    return values


class CameraControls:
    def __init__(self, index):
        self.graph = None
        self.interfaces = {}
        self.initialized = False
        self.error = None
        if os.name != "nt":
            self.error = "此平台未提供硬體影像調整"
            return
        try:
            import comtypes
            from ctypes import POINTER, c_long
            from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
            from pygrabber.dshow_graph import FilterGraph

            class IAMCameraControl(IUnknown):
                _iid_ = GUID("{C6E13370-30AC-11D0-A18C-00A0C9118956}")
                _methods_ = [
                    COMMETHOD([], HRESULT, "GetRange", (["in"], c_long, "property"),
                              *[(["out"], POINTER(c_long), name) for name in ("minimum", "maximum", "step", "default", "caps")]),
                    COMMETHOD([], HRESULT, "Set", (["in"], c_long, "property"), (["in"], c_long, "value"), (["in"], c_long, "flags")),
                    COMMETHOD([], HRESULT, "Get", (["in"], c_long, "property"), (["out"], POINTER(c_long), "value"), (["out"], POINTER(c_long), "flags")),
                ]

            class IAMVideoProcAmp(IUnknown):
                _iid_ = GUID("{C6E13360-30AC-11D0-A18C-00A0C9118956}")
                _methods_ = IAMCameraControl._methods_

            comtypes.CoInitialize()
            self.initialized = True
            self.graph = FilterGraph()
            self.graph.add_video_input_device(index)
            instance = self.graph.get_input_device().instance
            for name, interface in (("camera", IAMCameraControl), ("video", IAMVideoProcAmp)):
                try:
                    self.interfaces[name] = instance.QueryInterface(interface)
                except Exception:
                    pass
        except Exception as exc:
            self.error = f"無法讀取硬體參數：{exc}"
            self.close()

    def read(self):
        result = {}
        for key, (group, prop, label, advanced) in PROPERTIES.items():
            interface = self.interfaces.get(group)
            if interface is None:
                continue
            try:
                low, high, step, default, caps = map(int, interface.GetRange(prop))
                value, flags = map(int, interface.Get(prop))
                if low > high or step <= 0 or not low <= default <= high or not low <= value <= high:
                    continue
                result[key] = dict(label=label, advanced=advanced, min=low, max=high, step=step,
                                   default=default, value=value, auto=bool(flags & 1),
                                   supports_auto=bool(caps & 1), supports_manual=bool(caps & 2))
            except Exception:
                continue
        return result

    def apply(self, values):
        validate_values(values)
        current = self.read()
        # Validate the whole request before making any driver changes.
        for key, item in values.items():
            spec = current.get(key)
            if spec is None:
                raise ValueError(f"裝置不支援參數：{key}")
            value, auto = int(item["value"]), item["auto"]
            if not spec["supports_auto" if auto else "supports_manual"]:
                raise ValueError(f"{spec['label']}不支援此模式")
            if not auto and (not spec["min"] <= value <= spec["max"] or (value - spec["min"]) % spec["step"]):
                raise ValueError(f"{spec['label']}超出範圍或不符合步進值")
        errors = []
        for key, item in values.items():
            group, prop, label, _ = PROPERTIES[key]
            try:
                self.interfaces[group].Set(prop, int(item["value"]), 1 if item["auto"] else 2)
            except Exception:
                errors.append(f"{label}套用失敗")
        actual = self.read()
        for key, item in values.items():
            result = actual.get(key)
            if result is None or result["auto"] != item["auto"] or (not item["auto"] and result["value"] != item["value"]):
                errors.append(f"{PROPERTIES[key][2]}回讀值與要求不符")
        if errors:
            raise ValueError("；".join(errors))
        return actual

    def reset(self):
        return self.apply({key: {"value": spec["default"], "auto": spec["supports_auto"]}
                           for key, spec in self.read().items()})

    def close(self):
        self.interfaces.clear()
        try:
            if self.graph is not None:
                self.graph.remove_filters()
        finally:
            self.graph = None
            if self.initialized:
                import comtypes
                comtypes.CoUninitialize()
                self.initialized = False
