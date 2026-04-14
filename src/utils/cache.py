"""缓存层实现"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)


class Cache:
    """简单的文件缓存实现"""

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        """
        初始化缓存

        Args:
            cache_dir: 缓存目录
            ttl_hours: 缓存有效期（小时）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_path(self, key: str) -> Path:
        """根据key生成缓存文件路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存的值，如果不存在或已过期返回None
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        # 检查是否过期
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            # 缓存已过期，删除
            cache_path.unlink()
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"读取缓存失败: {key}, 错误: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存的值
        """
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"写入缓存失败: {key}, 错误: {e}")

    def delete(self, key: str) -> None:
        """删除缓存"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """
        获取缓存，如果不存在则计算并缓存

        Args:
            key: 缓存键
            compute_fn: 计算函数

        Returns:
            缓存的值或计算结果
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # 计算并缓存
        result = compute_fn()
        self.set(key, result)
        return result


def create_cache(cache_dir: Path, ttl_hours: int = 24) -> Cache:
    """创建缓存的工厂函数"""
    return Cache(cache_dir=cache_dir, ttl_hours=ttl_hours)


# 全局缓存实例
_global_cache: Optional[Cache] = None


def get_global_cache(cache_dir: Optional[Path] = None) -> Cache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        _global_cache = Cache(cache_dir)
    return _global_cache