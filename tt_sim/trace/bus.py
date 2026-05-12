"""Synchronous in-process pub/sub bus for simulator events.

Single global instance accessed via :func:`get_bus`. Consumers subscribe
per :class:`EventCategory`; the simulator publishes typed
:class:`~tt_sim.trace.events.Event` instances. The hot-path guard is
:meth:`EventBus.is_enabled` — a single attribute lookup that callers
should check before constructing event payloads.
"""

from collections.abc import Callable

from tt_sim.trace.events import Event, EventCategory

Subscriber = Callable[[Event], None]


class EventBus:
    __slots__ = ("_enabled", "_per_category", "_subscribers")

    def __init__(self):
        self._enabled: bool = False
        self._per_category: dict[EventCategory, bool] = {
            cat: True for cat in EventCategory
        }
        self._subscribers: dict[EventCategory, list[Subscriber]] = {
            cat: [] for cat in EventCategory
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def is_enabled(self, category: EventCategory) -> bool:
        return self._enabled and self._per_category[category]

    def set_category_enabled(self, category: EventCategory, value: bool):
        self._per_category[category] = value

    def subscribe(self, category: EventCategory, callback: Subscriber):
        self._subscribers[category].append(callback)

    def publish(self, event: Event):
        cat = event.CATEGORY
        if cat is None or not self._enabled or not self._per_category[cat]:
            return
        for sub in self._subscribers[cat]:
            sub(event)

    def reset(self):
        self._enabled = False
        for cat in self._subscribers:
            self._subscribers[cat] = []
            self._per_category[cat] = True


_BUS: EventBus | None = None


def get_bus() -> EventBus:
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS
