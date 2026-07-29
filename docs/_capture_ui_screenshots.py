"""Capture Nestling screenshots after child-dossier + purity fixes."""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:8015"


def shot(page, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    page.screenshot(path=str(p), full_page=True)
    print("SAVED", p.name, p.stat().st_size)


def pick_sex(page, form, value="male"):
    page.locator(f'{form} label.seg-opt:has(input[value="{value}"])').first.click()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for _ in range(40):
            try:
                if page.request.get(f"{BASE}/api/health").ok:
                    break
            except Exception:
                pass
            time.sleep(0.4)

        page.goto(f"{BASE}/", wait_until="domcontentloaded")
        page.evaluate("() => localStorage.clear()")
        page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(700)
        shot(page, "01_home.png")

        page.click("#lang-toggle")
        page.wait_for_timeout(500)
        shot(page, "01b_home_fa.png")
        page.click("#lang-toggle")

        # Create preterm child + growth so dossier has data
        page.goto(f"{BASE}/#/child", wait_until="domcontentloaded")
        page.fill('#child-form input[name="name"]', "Maya")
        pick_sex(page, "#child-form", "female")
        page.fill('#child-form input[name="gestational_age_weeks"]', "32")
        page.click("#child-submit")
        page.wait_for_timeout(1200)
        shot(page, "02_child_selected.png")

        page.goto(f"{BASE}/#/growth", wait_until="domcontentloaded")
        page.wait_for_timeout(400)
        # select child if present
        opts = page.locator("#growth-child option")
        if opts.count() > 1:
            page.select_option("#growth-child", index=1)
        pick_sex(page, "#growth-form", "female")
        page.fill('#growth-form input[name="weeks"]', "40")
        page.fill('#growth-form input[name="value"]', "2.9")
        page.click("#growth-submit")
        page.wait_for_selector("#growth-result:not([hidden])", timeout=30000)
        page.wait_for_timeout(800)
        shot(page, "06_growth.png")

        page.goto(f"{BASE}/#/child", wait_until="domcontentloaded")
        page.wait_for_timeout(800)
        # click first child to refresh dossier with growth
        btn = page.locator("#children-list .child-item").first
        if btn.count():
            btn.click()
            page.wait_for_timeout(1200)
        shot(page, "02b_child_dossier.png")

        page.goto(f"{BASE}/#/chat", wait_until="domcontentloaded")
        page.wait_for_selector("#chat-input")
        page.wait_for_timeout(500)
        shot(page, "03_chat_toolbar.png")
        # New chat + history buttons
        if page.locator("#btn-new-chat").count():
            page.click("#btn-new-chat")
            page.wait_for_timeout(600)
            shot(page, "03b_new_chat.png")
        if page.locator("#btn-chat-history").count():
            page.click("#btn-chat-history")
            page.wait_for_timeout(800)
            shot(page, "03c_chat_history.png")
            page.click("#btn-chat-history")

        page.fill("#chat-input", "show my child profile")
        page.click("#chat-send")
        page.wait_for_function(
            "() => /Maya|GA|growth/i.test(document.querySelector('#chat-thread').innerText)",
            timeout=60000,
        )
        page.wait_for_timeout(800)
        shot(page, "03d_chat_profile.png")

        page.fill("#chat-input", "show my child chart")
        page.click("#chat-send")
        page.wait_for_function(
            "() => /centile|WHO|INTERGROWTH|plotted|رسم/i.test(document.querySelector('#chat-thread').innerText)",
            timeout=90000,
        )
        page.wait_for_timeout(1200)
        shot(page, "05_chat_growth.png")

        page.fill("#chat-input", "when will my son talk?")
        page.click("#chat-send")
        page.wait_for_function(
            "() => /speech|words|talk|گفتار|کلمه/i.test(document.querySelector('#chat-thread').innerText)",
            timeout=90000,
        )
        page.wait_for_timeout(800)
        shot(page, "05b_chat_speech_no_chart.png")

        page.goto(f"{BASE}/#/screening", wait_until="domcontentloaded")
        page.wait_for_timeout(500)
        shot(page, "07_screening.png")
        page.locator("#asq-ages button, #asq-ages .chip").first.click()
        page.wait_for_selector("#screening-quiz:not([hidden])")
        page.wait_for_timeout(600)
        shot(page, "07b_asq.png")

        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(f"{BASE}/#/", wait_until="domcontentloaded")
        page.wait_for_timeout(500)
        shot(page, "08_home_mobile.png")
        browser.close()
    print("FILES", sorted(x.name for x in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
