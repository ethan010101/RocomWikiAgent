"""
洛克王国 BWIKI 页面枚举与筛选逻辑，对齐 import_lkwiki.py / import_pets.py。
- 列表：api.php?action=query&list=allpages
- 词条 URL：https://wiki.biligame.com/rocom/{quote(title)}
"""
from __future__ import annotations

import os
import time
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKI_BASE_URL = os.getenv("WIKI_BASE_URL", "https://wiki.biligame.com/rocom").rstrip("/")
WIKI_API_URL = os.getenv("WIKI_API_URL", f"{WIKI_BASE_URL}/api.php")
WIKI_HTTP_RETRIES = int(os.getenv("WIKI_HTTP_RETRIES", "5"))
WIKI_RETRY_BACKOFF = float(os.getenv("WIKI_RETRY_BACKOFF", "0.7"))
WIKI_LIST_RETRY = int(os.getenv("WIKI_LIST_RETRY", "4"))


def wiki_http_session() -> requests.Session:
    """带重试与连接池，缓解 RemoteDisconnected / 空响应。"""
    s = requests.Session()
    retry = Retry(
        total=WIKI_HTTP_RETRIES,
        connect=WIKI_HTTP_RETRIES,
        read=WIKI_HTTP_RETRIES,
        backoff_factor=WIKI_RETRY_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )
    return s

# ---------- import_pets.py：宠物筛选 ----------
PET_KEYWORDS = [
    "宠物",
    "精灵",
    "火系",
    "水系",
    "草系",
    "冰系",
    "电系",
    "翼系",
    "武系",
    "毒系",
    "土系",
    "石系",
    "萌系",
    "虫系",
    "恶魔系",
    "幽灵系",
    "机械系",
    "龙系",
    "光系",
    "普通系",
]

KNOWN_PETS = [
    "火花",
    "焰火",
    "火神",
    "烈火战神",
    "水蓝蓝",
    "波波拉",
    "水灵",
    "圣水守护",
    "喵喵",
    "喵呜",
    "魔力猫",
    "武斗酷猫",
    "音速犬",
    "风暴战犬",
    "蹦蹦种子",
    "蹦蹦草",
    "蹦蹦花",
    "蹦神幻影",
    "火苗龙",
    "烈焰龙",
    "烈火飞龙",
    "板板壳",
    "咔咔壳",
    "水泡壳",
    "毛毛",
    "爬爬虫",
    "独角虫",
    "地鼠",
    "遁鼠",
    "白发懒人",
    "动力猿",
    "瞌睡王",
    "蒲公英",
    "大蒲公英",
    "奇丽草",
    "奇丽花",
    "奇丽果",
    "小天马",
    "独角兽",
    "白金独角兽",
    "黄金独角兽",
    "小钻鼠",
    "大钻鼠",
    "小蜂猴",
    "大头蜂猴",
    "小地狱犬",
    "地狱三头犬",
    "小甲基丸",
    "甲基丸",
    "小翼龙",
    "翼龙",
    "翼龙王子",
    "小机械贝贝",
    "机械贝贝",
    "机械阿蛮",
    "小帕蔻",
    "帕蔻",
    "小星光",
    "星光狮",
    "小灵桃",
    "灵桃子",
    "小曼陀罗猪",
    "曼陀罗猪神",
    "小奶牛",
    "斑点奶牛",
    "阿布",
    "超能阿布",
    "圣龙阿布",
    "迪莫",
    "圣光迪莫",
    "皇家圣光迪莫",
    "麻球",
    "绿灵麻球",
    "超凡麻球",
    "蔴球",
    "绿灵蔴球",
    "超凡蔴球",
    "圣藤草王",
    "灵蔓草王",
    "冰龙王",
    "暗影冰龙王",
    "萌之王者",
    "萌宝宝",
    "上古战龙",
    "劫影龙皇",
    "皇家狮鹫",
    "超级皇家狮鹫",
    "小光",
    "大光",
    "流光",
    "幽冥紫灯",
    "幽灵灯",
    "小天使",
    "大天使",
    "觉醒大天使",
    "小恶魔",
    "恶魔男爵",
    "小青龙",
    "青龙",
    "神圣青龙",
    "小朱雀",
    "朱雀",
    "菲尼克斯",
    "小玄武",
    "玄武",
    "北圣兽元紫",
    "小白虎",
    "白虎",
    "西圣兽白鄂",
]


def is_pet_related_page(title: str) -> bool:
    if title.startswith(("File:", "Category:", "Template:", "User:", "Help:", "MediaWiki:")):
        return False
    skip_keywords = ["首页", "更新", "公告", "攻略", "活动", "任务", "地图", "道具", "装备", "技能石"]
    for kw in skip_keywords:
        if kw in title and not any(pk in title for pk in PET_KEYWORDS):
            return False
    for keyword in PET_KEYWORDS:
        if keyword in title:
            return True
    for pet in KNOWN_PETS:
        if pet in title:
            return True
    return False


def is_lkwiki_page(title: str) -> bool:
    """import_lkwiki.py：含命名空间时，仅保留指定前缀。"""
    if ":" in title and not title.startswith(("宠物", "技能", "道具", "地图", "任务")):
        return False
    return True


def title_to_article_url(title: str) -> str:
    return f"{WIKI_BASE_URL}/{quote(title.replace(' ', '_'))}"


def title_to_index_php_url(title: str) -> str:
    return f"{WIKI_BASE_URL}/index.php?{urlencode({'title': title.replace(' ', '_')})}"


def title_fetch_url_candidates(title: str, preferred_article_url: str | None = None) -> list[str]:
    """同一词条多种 URL，BWIKI 上 path 与 index.php 表现可能不同。"""
    seen: set[str] = set()
    out: list[str] = []
    for u in (
        preferred_article_url,
        title_to_article_url(title),
        title_to_index_php_url(title),
    ):
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_all_titles(session: requests.Session, timeout: int = 30) -> list[str]:
    titles: list[str] = []
    apcontinue = None
    while True:
        params: dict = {
            "action": "query",
            "list": "allpages",
            "aplimit": 500,
            "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        last_err: Exception | None = None
        data = None
        for attempt in range(WIKI_LIST_RETRY):
            try:
                resp = session.get(WIKI_API_URL, params=params, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(WIKI_RETRY_BACKOFF * (attempt + 1))
        if last_err is not None:
            raise last_err
        assert data is not None
        if "query" in data and "allpages" in data["query"]:
            for page in data["query"]["allpages"]:
                titles.append(page["title"])
        if "continue" in data and "apcontinue" in data["continue"]:
            apcontinue = data["continue"]["apcontinue"]
            time.sleep(0.5)
        else:
            break
    return titles


def get_core_pet_pages() -> list[dict]:
    """import_pets.py 备选核心列表"""
    return [
        {"title": "宠物图鉴", "url": f"{WIKI_BASE_URL}/%E5%AE%A0%E7%89%A9%E5%9B%BE%E9%89%B4"},
        {"title": "火花", "url": f"{WIKI_BASE_URL}/%E7%81%AB%E8%8A%B1"},
        {"title": "焰火", "url": f"{WIKI_BASE_URL}/%E7%84%B0%E7%81%AB"},
        {"title": "火神", "url": f"{WIKI_BASE_URL}/%E7%81%AB%E7%A5%9E"},
        {"title": "水蓝蓝", "url": f"{WIKI_BASE_URL}/%E6%B0%B4%E8%93%9D%E8%93%9D"},
        {"title": "波波拉", "url": f"{WIKI_BASE_URL}/%E6%B3%A2%E6%B3%A2%E6%8B%89"},
        {"title": "水灵", "url": f"{WIKI_BASE_URL}/%E6%B0%B4%E7%81%B5"},
        {"title": "喵喵", "url": f"{WIKI_BASE_URL}/%E5%96%B5%E5%96%B5"},
        {"title": "喵呜", "url": f"{WIKI_BASE_URL}/%E5%96%B5%E5%91%9C"},
        {"title": "魔力猫", "url": f"{WIKI_BASE_URL}/%E9%AD%94%E5%8A%9B%E7%8C%AB"},
        {"title": "音速犬", "url": f"{WIKI_BASE_URL}/%E9%9F%B3%E9%80%9F%E7%8A%AC"},
        {"title": "瞌睡王", "url": f"{WIKI_BASE_URL}/%E7%9E%8C%E7%9D%A1%E7%8E%8B"},
        {"title": "皇家狮鹫", "url": f"{WIKI_BASE_URL}/%E7%9A%87%E5%AE%B6%E7%8B%AE%E9%B9%AB"},
        {"title": "迪莫", "url": f"{WIKI_BASE_URL}/%E8%BF%AA%E8%8E%AB"},
        {"title": "阿布", "url": f"{WIKI_BASE_URL}/%E9%98%BF%E5%B8%83"},
        {"title": "冰龙王", "url": f"{WIKI_BASE_URL}/%E5%86%B0%E9%BE%99%E7%8E%8B"},
        {"title": "萌之王者", "url": f"{WIKI_BASE_URL}/%E8%90%8C%E4%B9%8B%E7%8E%8B%E8%80%85"},
        {"title": "上古战龙", "url": f"{WIKI_BASE_URL}/%E4%B8%8A%E5%8F%A4%E6%88%98%E9%BE%99"},
        {"title": "火系宠物", "url": f"{WIKI_BASE_URL}/%E7%81%AB%E7%B3%BB"},
        {"title": "水系宠物", "url": f"{WIKI_BASE_URL}/%E6%B0%B4%E7%B3%BB"},
        {"title": "草系宠物", "url": f"{WIKI_BASE_URL}/%E8%8D%89%E7%B3%BB"},
        {"title": "冰系宠物", "url": f"{WIKI_BASE_URL}/%E5%86%B0%E7%B3%BB"},
    ]


def get_important_pages() -> list[dict]:
    """import_lkwiki.py 备选重要页面"""
    return [
        {"title": "首页", "url": f"{WIKI_BASE_URL}/%E9%A6%96%E9%A1%B5"},
        {"title": "宠物图鉴", "url": f"{WIKI_BASE_URL}/%E5%AE%A0%E7%89%A9%E5%9B%BE%E9%89%B4"},
        {"title": "火系宠物", "url": f"{WIKI_BASE_URL}/%E7%81%AB%E7%B3%BB"},
        {"title": "水系宠物", "url": f"{WIKI_BASE_URL}/%E6%B0%B4%E7%B3%BB"},
        {"title": "草系宠物", "url": f"{WIKI_BASE_URL}/%E8%8D%89%E7%B3%BB"},
        {"title": "火神", "url": f"{WIKI_BASE_URL}/%E7%81%AB%E7%A5%9E"},
        {"title": "水蓝蓝", "url": f"{WIKI_BASE_URL}/%E6%B0%B4%E8%93%9D%E8%93%9D"},
        {"title": "喵喵", "url": f"{WIKI_BASE_URL}/%E5%96%B5%E5%96%B5"},
        {"title": "属性相克", "url": f"{WIKI_BASE_URL}/%E5%B1%9E%E6%80%A7%E7%9B%B8%E5%85%8B"},
        {"title": "技能", "url": f"{WIKI_BASE_URL}/%E6%8A%80%E8%83%BD"},
        {"title": "地图", "url": f"{WIKI_BASE_URL}/%E5%9C%B0%E5%9B%BE"},
        {"title": "天空城", "url": f"{WIKI_BASE_URL}/%E5%A4%A9%E7%A9%BA%E5%9F%8E"},
        {"title": "人鱼湾", "url": f"{WIKI_BASE_URL}/%E4%BA%BA%E9%B1%BC%E6%B9%BE"},
        {"title": "任务", "url": f"{WIKI_BASE_URL}/%E4%BB%BB%E5%8A%A1"},
        {"title": "对战", "url": f"{WIKI_BASE_URL}/%E5%AF%B9%E6%88%98"},
        {"title": "进化", "url": f"{WIKI_BASE_URL}/%E8%BF%9B%E5%8C%96"},
    ]


def list_pages_for_kb(mode: str, session: requests.Session) -> list[dict]:
    """
    mode: 'all' -> import_lkwiki 规则；'pets' -> import_pets 规则。
    API 失败或结果为空时使用与 Coze 脚本一致的备选 URL 列表。
    """
    mode = (mode or "all").strip().lower()
    try:
        titles = fetch_all_titles(session)
        picked: list[dict] = []
        for t in titles:
            if mode == "pets":
                if not is_pet_related_page(t):
                    continue
            else:
                if not is_lkwiki_page(t):
                    continue
            picked.append({"title": t, "url": title_to_article_url(t)})
        if picked:
            return picked
    except Exception as e:
        print(f"[wiki] API 列表失败，使用备选页面: {e}")
    if mode == "pets":
        return get_core_pet_pages()
    return get_important_pages()
