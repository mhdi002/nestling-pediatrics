from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8011"
OUT = Path("docs/screenshots")


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page(viewport={"width": 1280, "height": 900})
        page.goto(BASE + "/", wait_until="domcontentloaded")
        page.evaluate("() => localStorage.clear()")
        page.goto(BASE + "/#/chat", wait_until="domcontentloaded")
        page.fill("#chat-input", "boy weight 3.2 kg at 40 weeks")
        page.click("#chat-send")
        page.wait_for_function(
            "() => /centile|chart below/i.test(document.querySelector('#chat-thread').innerText)",
            timeout=90000,
        )
        page.wait_for_timeout(2000)
        imgs = page.locator("#chat-thread img.overlay-img").count()
        print("overlay_imgs", imgs)
        page.screenshot(path=str(OUT / "05_chat_growth.png"), full_page=True)
        page.goto(BASE + "/#/screening", wait_until="domcontentloaded")
        page.locator("#asq-ages button, #asq-ages .chip").first.click()
        page.wait_for_selector("#screening-quiz:not([hidden])")
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "07b_asq_quiz.png"), full_page=True)
        page.screenshot(path=str(OUT / "07c_asq_in_progress.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(BASE + "/#/", wait_until="domcontentloaded")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "08_home_mobile.png"), full_page=True)
        b.close()
    print("ok")


if __name__ == "__main__":
    main()
