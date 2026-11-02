
import msgpack

class _OrmsgpackConsts:
    # 核心选项常量
    OPT_NAIVE_UTC = 1 << 0
    OPT_NON_STR_KEYS = 1 << 1
    OPT_OMIT_MICROSECONDS = 1 << 2
    OPT_PASSTHROUGH_BIG_INT = 1 << 3
    OPT_PASSTHROUGH_DATACLASS = 1 << 4
    OPT_PASSTHROUGH_DATETIME = 1 << 5
    OPT_PASSTHROUGH_ENUM = 1 << 6
    OPT_PASSTHROUGH_SUBCLASS = 1 << 7
    OPT_PASSTHROUGH_TUPLE = 1 << 8
    OPT_PASSTHROUGH_UUID = 1 << 9
    OPT_SERIALIZE_NUMPY = 1 << 10
    OPT_SERIALIZE_PYDANTIC = 1 << 11
    OPT_SORT_KEYS = 1 << 12
    OPT_UTC_Z = 1 << 13
    OPT_DATETIME_AS_TIMESTAMP_EXT = 1 << 14
    OPT_REPLACE_SURROGATES = 1 << 14

    class PackError(Exception):
        pass

    class UnpackError(Exception):
        pass

    class Ext:
        def __init__(self, code, data):
            self.code = code
            self.data = data

    @classmethod
    def packb(cls, data, **kwargs):
        """使用 msgpack 进行序列化"""

        def _default(obj):
            # 特殊处理 Interrupt 对象
            if obj.__class__.__name__ == 'Interrupt':
                return {
                    "__interrupt__": True,
                    "value": getattr(obj, 'value', None),
                    "id": getattr(obj, 'id', None)
                }

            # 处理 Ext 类型
            if isinstance(obj, cls.Ext):
                return {"__ext_code__": obj.code, "__ext_data__": obj.data}

            # 处理 LangChain 消息对象（保持原始结构）
            if hasattr(obj, 'content') and hasattr(obj, 'type'):
                result = {
                    "content": obj.content,
                    "type": obj.type,
                    "id": getattr(obj, 'id', None),
                    "name": getattr(obj, 'name', None),
                    "additional_kwargs": getattr(obj, 'additional_kwargs', {}),
                    "response_metadata": getattr(obj, 'response_metadata', {}),
                    "tool_calls": getattr(obj, 'tool_calls', []),
                    "invalid_tool_calls": getattr(obj, 'invalid_tool_calls', []),
                    "usage_metadata": getattr(obj, 'usage_metadata', None),
                }
                # 添加 tool_call_id 用于 ToolMessage，如果存在的话
                if hasattr(obj, 'tool_call_id') and obj.tool_call_id is not None:
                    result["tool_call_id"] = obj.tool_call_id
                return result

            # 处理 datetime
            if hasattr(obj, 'isoformat') and hasattr(obj, 'strftime'):
                return {"__datetime__": obj.isoformat()}

            # 处理 UUID
            if hasattr(obj, 'hex') and hasattr(obj, 'bytes'):
                return {"__uuid__": str(obj)}

            # 处理 dataclass
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    "__dataclass__": obj.__class__.__name__,
                    "__module__": obj.__class__.__module__,
                    "fields": {k: getattr(obj, k) for k in obj.__dataclass_fields__}
                }

            # 处理有 __dict__ 的对象
            if hasattr(obj, '__dict__'):
                return {
                    "__class__": obj.__class__.__name__,
                    "__module__": obj.__class__.__module__,
                    "__dict__": obj.__dict__
                }

            # 最后的回退
            return str(obj)

        return msgpack.packb(data, default=_default, use_bin_type=True)

    @classmethod
    def unpackb(cls, data, **kwargs):
        """使用 msgpack 进行反序列化"""

        def _object_hook(obj):
            if not isinstance(obj, dict):
                return obj

            # 还原 Interrupt 对象 - 更严格的判断
            if (obj.get("__interrupt__") is True and
                    "value" in obj and
                    "id" in obj and
                    len(obj) == 3):
                try:
                    from langgraph.types import Interrupt
                    return Interrupt(
                        value=obj.get("value"),
                        id=obj.get("id")
                    )
                except ImportError:
                    return obj

            # 还原 LangChain 消息对象
            if ("content" in obj and
                    "type" in obj and
                    obj.get("type") in ["human", "ai", "system", "tool"]):
                msg_type = obj.get("type")
                try:
                    from langchain_core.messages import (
                        HumanMessage, AIMessage, SystemMessage, ToolMessage
                    )

                    if msg_type == "human":
                        return HumanMessage(
                            content=obj["content"],
                            id=obj.get("id"),
                            name=obj.get("name"),
                            additional_kwargs=obj.get("additional_kwargs", {}),
                            response_metadata=obj.get("response_metadata", {})
                        )
                    elif msg_type == "ai":
                        return AIMessage(
                            content=obj["content"],
                            id=obj.get("id"),
                            name=obj.get("name"),
                            additional_kwargs=obj.get("additional_kwargs", {}),
                            response_metadata=obj.get("response_metadata", {}),
                            tool_calls=obj.get("tool_calls", []),
                            invalid_tool_calls=obj.get("invalid_tool_calls", []),
                            usage_metadata=obj.get("usage_metadata")
                        )
                    elif msg_type == "system":
                        return SystemMessage(
                            content=obj["content"],
                            id=obj.get("id"),
                            name=obj.get("name"),
                            additional_kwargs=obj.get("additional_kwargs", {}),
                            response_metadata=obj.get("response_metadata", {})
                        )
                    elif msg_type == "tool":
                        # 处理 ToolMessage，tool_call_id 可能为 None
                        tool_call_id = obj.get("tool_call_id")
                        # 如果 tool_call_id 为 None，提供一个默认值
                        if tool_call_id is None:
                            tool_call_id = obj.get("id", "unknown_tool_call")

                        return ToolMessage(
                            content=obj["content"],
                            tool_call_id=tool_call_id,
                            id=obj.get("id"),
                            name=obj.get("name"),
                            additional_kwargs=obj.get("additional_kwargs", {}),
                            response_metadata=obj.get("response_metadata", {})
                        )
                except (ImportError, ValueError, TypeError) as e:
                    # 如果创建消息对象失败，返回原始字典
                    print(f"Warning: Failed to create {msg_type} message: {e}")
                    return obj

            # 还原 Ext 类型
            if ("__ext_code__" in obj and
                    "__ext_data__" in obj and
                    len(obj) == 2):
                return cls.Ext(obj["__ext_code__"], obj["__ext_data__"])

            # 还原 datetime
            if ("__datetime__" in obj and
                    len(obj) == 1):
                from datetime import datetime
                try:
                    return datetime.fromisoformat(obj["__datetime__"])
                except (ValueError, TypeError):
                    pass

            # 还原 UUID
            if ("__uuid__" in obj and
                    len(obj) == 1):
                import uuid
                try:
                    return uuid.UUID(obj["__uuid__"])
                except (ValueError, AttributeError):
                    pass

            # 还原 dataclass
            if ("__dataclass__" in obj and
                    "__module__" in obj and
                    "fields" in obj):
                try:
                    import importlib
                    module = importlib.import_module(obj["__module__"])
                    cls_ = getattr(module, obj["__dataclass__"])
                    return cls_(**obj.get("fields", {}))
                except (ImportError, AttributeError, TypeError):
                    pass

            # 还原普通对象
            if ("__class__" in obj and
                    "__module__" in obj and
                    "__dict__" in obj and
                    len(obj) == 3):
                try:
                    import importlib
                    module = importlib.import_module(obj["__module__"])
                    cls_ = getattr(module, obj["__class__"])
                    instance = cls_.__new__(cls_)
                    instance.__dict__.update(obj["__dict__"])
                    return instance
                except (ImportError, AttributeError, TypeError):
                    pass

            # 如果都不匹配，返回原始字典
            return obj

        return msgpack.unpackb(data, object_hook=_object_hook, raw=False)

