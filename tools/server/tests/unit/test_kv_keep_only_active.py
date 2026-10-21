import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield
    os.environ.pop("LLAMA_SERVER_LAZY_IDLE_SLOTS", None)


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)

# the tiny test model has a 512-token unified cache for both slots, so these two prompts
# together do not fit: the second one has to make room in the KV cache
PROMPT_A = LONG_PROMPT * 2
PROMPT_B = "Meanwhile, " + LONG_PROMPT * 2


# by default an idle slot keeps its KV; the KV-full path saves it to cache-ram before clearing it
def test_idle_slots_stay_resident():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT_A,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # the idle slot is not saved and cleared when another task starts
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # overflow the unified KV cache: the idle slot is saved to cache-ram and only then cleared
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT_B,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "making room in the KV cache: saving and clearing idle slot" in log.drain()

    # the cleared conversation comes back from cache-ram instead of being recomputed
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT_A,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# LLAMA_SERVER_LAZY_IDLE_SLOTS=0: the eager behaviour (save and clear every idle slot on a new task)
def test_idle_slots_eager_kill_switch():
    global server
    os.environ["LLAMA_SERVER_LAZY_IDLE_SLOTS"] = "0"
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # launching slot 1 saves and clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # re-send the same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()
    assert "making room in the KV cache" not in log.drain()
