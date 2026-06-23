import os
import gin
import asyncio
import jinja2
import json
import re
import sys
sys.path.append('.')

from pathlib import Path
from typing import Dict, List, Any, Optional
from api.utils.key_operator import ApiKeyCycler
from api.CustomizeChatGeneratorService import CustomizeChatGenerator
from api.utils.string_operator import json_fix
from api.utils.url_operator import compress_and_convert_base64, compress_url
from dotenv import load_dotenv


load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ---------------------------------------------------------------------------
# XHS Copywriting Prompt
# ---------------------------------------------------------------------------

XHS_COPYWRITING_PROMPT = """你是一个拥有2000w粉丝的小红书爆款写作专家，同时拥有消费心理学+市场营销双PhD。
你是小红书的重度用户，拥有卓越的互联网网感。你的语气和写作风格非常小红书化。
你只在中文互联网语境下创作，使用自然富有网感的中文。

现在请你根据以下内容，创作一篇小红书爆款笔记。

## 内容摘要
{summary}

## 创作要求

### 标题（5个备选）
- 每个标题字数限制在20以内
- 含适当的emoji表情
- 使用爆炸词（带有强烈情感倾向且能引起用户共鸣的词语）
- 制造好奇心或共鸣感

### 正文（1篇）
- 每个段落都含有适当的emoji表情（同一emoji不重复出现）
- 开头要有hook，抓住注意力
- 内容要干货满满，有信息增量
- 语言口语化、亲切自然
- 适当使用"！""～""…"等语气词
- 文末附上合适的SEO标签（#开头）

## 输出格式（严格JSON）
```json
{{
  "titles": [
    "标题1（含emoji）",
    "标题2（含emoji）",
    "标题3（含emoji）",
    "标题4（含emoji）",
    "标题5（含emoji）"
  ],
  "body": "正文内容（含emoji和段落分隔）",
  "tags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"]
}}
```
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _body_to_html(body: str) -> str:
    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    return "\n".join(f"        <p>{_escape(p)}</p>" for p in paragraphs)


# ---------------------------------------------------------------------------
# XHS HTML Template
# ---------------------------------------------------------------------------

_XHS_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>{title}</title>
<style>
  :root {{
    --xhs-red: #ff2442;
    --xhs-bg: #fafafa;
    --xhs-card: #ffffff;
    --xhs-text: #333333;
    --xhs-sub: #999999;
    --xhs-tag: #f0f0f0;
    --radius: 12px;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html {{ font-size: 16px; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif;
    background: var(--xhs-bg);
    color: var(--xhs-text);
    max-width: 480px;
    margin: 0 auto;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }}

  /* ---- Carousel ---- */
  .carousel-wrap {{
    position: relative;
    width: 100%;
    background: #000;
    overflow: hidden;
  }}
  .carousel {{
    display: flex;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
  }}
  .carousel::-webkit-scrollbar {{ display: none; }}
  .slide {{
    flex: 0 0 100%;
    scroll-snap-align: start;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .slide img {{
    width: 100%;
    height: auto;
    display: block;
    object-fit: contain;
  }}

  /* ---- Counter ---- */
  .counter {{
    position: absolute;
    top: 14px;
    right: 14px;
    background: rgba(0,0,0,.55);
    color: #fff;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 20px;
    pointer-events: none;
    z-index: 5;
  }}

  /* ---- Dots ---- */
  .dots {{
    display: flex;
    justify-content: center;
    gap: 6px;
    padding: 10px 0 6px;
    background: var(--xhs-card);
  }}
  .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #ddd;
    transition: all .3s;
    cursor: pointer;
  }}
  .dot.active {{
    background: var(--xhs-red);
    width: 18px;
    border-radius: 3px;
  }}

  /* ---- Arrows ---- */
  .arrow {{
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 36px; height: 36px;
    background: rgba(255,255,255,.75);
    border: none;
    border-radius: 50%;
    font-size: 18px;
    color: #333;
    cursor: pointer;
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity .2s;
  }}
  .carousel-wrap:hover .arrow {{ opacity: 1; }}
  .arrow-left {{ left: 10px; }}
  .arrow-right {{ right: 10px; }}

  /* ---- Content card ---- */
  .content {{
    background: var(--xhs-card);
    padding: 16px 18px 24px;
  }}
  .main-title {{
    font-size: 18px;
    font-weight: 700;
    line-height: 1.5;
    margin-bottom: 12px;
    color: var(--xhs-text);
  }}
  .body p {{
    font-size: 15px;
    line-height: 1.75;
    margin-bottom: 10px;
    color: #444;
  }}
  .tags {{
    margin-top: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .tag {{
    display: inline-block;
    font-size: 13px;
    color: #4a90d9;
    background: var(--xhs-tag);
    padding: 4px 12px;
    border-radius: 20px;
  }}

  /* ---- Alt titles ---- */
  .alt-titles {{
    padding: 16px 18px;
    background: var(--xhs-card);
    margin-top: 8px;
    border-radius: var(--radius);
  }}
  .alt-titles-label {{
    font-size: 13px;
    color: var(--xhs-sub);
    margin-bottom: 8px;
  }}
  .alt-title {{
    font-size: 14px;
    color: #555;
    padding: 6px 0;
    border-bottom: 1px solid #f5f5f5;
  }}
  .alt-title:last-child {{ border-bottom: none; }}

  /* ---- Footer ---- */
  .footer {{
    text-align: center;
    padding: 20px;
    font-size: 12px;
    color: var(--xhs-sub);
  }}
</style>
</head>
<body>

  <div class="carousel-wrap">
    <div class="counter"><span id="cur">1</span>/{num_images}</div>
    <button class="arrow arrow-left" onclick="go(-1)">&#8249;</button>
    <button class="arrow arrow-right" onclick="go(1)">&#8250;</button>
    <div class="carousel" id="carousel">
{slides_html}
    </div>
  </div>

  <div class="dots" id="dots">
{dots_html}
  </div>

  <div class="content">
    <div class="main-title">{title}</div>
    <div class="body">
{body_html}
    </div>
    <div class="tags">
      {tags_html}
    </div>
  </div>

  <div class="alt-titles">
    <div class="alt-titles-label">&#128204; &#22791;&#36873;&#26631;&#39064;</div>
{titles_html}
  </div>

  <div class="footer">Generated by PosterGen</div>

<script>
(function() {{
  const carousel = document.getElementById('carousel');
  const dots = document.querySelectorAll('.dot');
  const counter = document.getElementById('cur');
  const total = {num_images};
  let idx = 0;

  function update(i) {{
    idx = Math.max(0, Math.min(i, total - 1));
    carousel.children[idx].scrollIntoView({{ behavior: 'smooth', block: 'nearest', inline: 'start' }});
    dots.forEach((d, j) => d.classList.toggle('active', j === idx));
    counter.textContent = idx + 1;
  }}

  carousel.addEventListener('scroll', function() {{
    const w = carousel.offsetWidth;
    const newIdx = Math.round(carousel.scrollLeft / w);
    if (newIdx !== idx && newIdx >= 0 && newIdx < total) {{
      idx = newIdx;
      dots.forEach((d, j) => d.classList.toggle('active', j === idx));
      counter.textContent = idx + 1;
    }}
  }});

  dots.forEach(d => d.addEventListener('click', function() {{
    update(parseInt(this.dataset.idx));
  }}));

  window.go = function(dir) {{ update(idx + dir); }};

  document.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowLeft') update(idx - 1);
    if (e.key === 'ArrowRight') update(idx + 1);
  }});
}})();
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@gin.configurable()
class SlideRender:
    def __init__(
        self,
        model_name: str,
        use_zh: bool = True,
        max_retries: int = 5,
        retry_delay: int = 3,
        max_concurrent: int = 50,
        use_customize_url: bool = False,
        customize_url: str = "",
        max_summary_len: int = 6000,
        image_key_type: str = "url",
        max_num_images: int = 14,
        use_api_key: bool = True,
        system_template_dir: str = "./template",
        system_template_en_file: str = "SlideRender_EN.jinja2",
        system_template_zh_file: str = "SlideRender_ZH.jinja2",
        html_template_en_file: str = "HTMLRender_EN.jinja2",
        html_template_zh_file: str = "HTMLRender_ZH.jinja2",
        render_template_dir: str = "./gallery",
        render_template_en_file: str = "KIMIVertical.txt",
        render_template_zh_file: str = "KIMIVertical.txt",
        render_with_image: bool = True,
    ):
        self.model = CustomizeChatGenerator(
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_customize_url=use_customize_url,
            customize_url=customize_url,
            use_api_key=use_api_key,
        )

        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(system_template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.jinja_file = system_template_en_file if not use_zh else system_template_zh_file
        self.html_jinja_file = html_template_en_file if not use_zh else html_template_zh_file
        self.render_file = os.path.join(render_template_dir, render_template_en_file) if not use_zh else os.path.join(render_template_dir, render_template_zh_file)
        self.use_zh = use_zh
        self.render_with_image = render_with_image
        self.max_summary_len = max_summary_len

    def get_system_prompt(self, summary: str = "") -> str:
        template = self.jinja_env.get_template(self.jinja_file)
        return template.render(summary=summary)

    def check_func(self, response: str) -> Dict[str, Any]:
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            data = {
                "titles": ["📝 精彩内容速递"],
                "body": response,
                "tags": ["#知识分享"],
            }

        return {
            "titles": data.get("titles", ["📝 精彩内容速递"]),
            "body": data.get("body", ""),
            "tags": data.get("tags", []),
        }

    async def post_request(
        self,
        query_text: str,
        report: str,
    ) -> Dict[str, Any]:
        summary = report[:self.max_summary_len]
        prompt_text = self.get_system_prompt(summary=summary)

        if "gemini" in self.model.model_name:
            system_prompt = [{"text": prompt_text}]
            user_prompt = []

            if self.use_zh:
                _user_prompt = f"""
# 用户提问
{query_text}

# 报告内容
{summary}
"""
            else:
                _user_prompt = f"""
# User Question
{query_text}

# Report
{summary}
"""
            user_prompt.append({"text": _user_prompt})

            response = await self.model.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=lambda r: r,
                return_cot=False,
            )
        else:
            raise ValueError(f"Unsupported {self.model.model_name}")

        return self.check_func(response)

    def build_xhs_html(
        self,
        image_paths: List[str],
        xhs_copy: Dict[str, Any],
        output_dir: str,
        title_index: int = 0,
    ) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        html_path = out / "xhs_post.html"

        rel_images = []
        for p in image_paths:
            try:
                rel = os.path.relpath(p, str(out))
            except ValueError:
                rel = p
            rel_images.append(rel)

        titles = xhs_copy.get("titles", ["📝"])
        title = titles[title_index % max(len(titles), 1)]
        body = xhs_copy.get("body", "")
        tags = xhs_copy.get("tags", [])
        num_images = len(rel_images)

        body_html = _body_to_html(body)
        tags_html = " ".join(
            f'<span class="tag">{t}</span>' for t in tags
        )
        titles_html = "\n".join(
            f'            <div class="alt-title">{t}</div>' for t in titles
        )
        slides_html = "\n".join(
            f'          <div class="slide"><img src="{img}" alt="slide {i+1}"></div>'
            for i, img in enumerate(rel_images)
        )
        dots_html = "\n".join(
            f'          <span class="dot{" active" if i == 0 else ""}" data-idx="{i}"></span>'
            for i in range(num_images)
        )

        html = _XHS_TEMPLATE.format(
            title=_escape(title),
            slides_html=slides_html,
            dots_html=dots_html,
            num_images=num_images,
            body_html=body_html,
            tags_html=tags_html,
            titles_html=titles_html,
        )

        html_path.write_text(html, encoding="utf-8")
        print(f"[SlideRender] XHS HTML saved: {html_path}")
        return str(html_path)

    async def act(
        self,
        input_dict: Dict,
        input_key: str = "rendered_report",
    ) -> Dict:
        query_text = input_dict.get("query_text", "") or input_dict.get("query", "")
        report = input_dict[input_key]

        xhs_copy = await self.post_request(
            query_text=query_text,
            report=report,
        )
        input_dict["xhs_copy"] = xhs_copy

        image_paths = input_dict.get("xhs_slide_images", [])
        output_dir = input_dict.get("output_dir", "")
        if image_paths and output_dir:
            html_path = self.build_xhs_html(
                image_paths=image_paths,
                xhs_copy=xhs_copy,
                output_dir=output_dir,
            )
            input_dict["xhs_html_path"] = html_path

        return input_dict


if __name__ == "__main__":
    REPORT = "量子计算正在改变世界。MIT、斯坦福、清华等顶尖高校在拓扑量子计算、后量子密码学等领域取得了重大突破。"

    async def main():
        service = SlideRender(
            model_name="gemini-3-flash",
        )
        input_dict = {
            "query_text": "量子计算最新进展",
            "rendered_report": REPORT,
        }
        input_dict = await service.act(input_dict=input_dict)
        print(json.dumps(input_dict.get("xhs_copy", {}), ensure_ascii=False, indent=2))

    asyncio.run(main())
