#!/usr/bin/env node

const fs = require('node:fs/promises');
const path = require('node:path');

let puppeteer;
try {
  puppeteer = require('puppeteer');
} catch (err) {
  console.error('Puppeteer is not installed. Run "npm install puppeteer" in the project root.');
  process.exit(1);
}

const args = process.argv.slice(2);

if (args.length === 0) {
  console.error('Usage: node tools/capture_sections.js <url> [--output <dir>] [--clip 1000]');
  process.exit(1);
}

const urlArg = args[0];
let outputDir = 'logs/screenshots';
let clipHeight = 1000;

for (let i = 1; i < args.length; i += 1) {
  const flag = args[i];
  if (flag === '--output' && args[i + 1]) {
    outputDir = args[i + 1];
    i += 1;
  } else if (flag === '--clip' && args[i + 1]) {
    clipHeight = parseInt(args[i + 1], 10) || clipHeight;
    i += 1;
  }
}

function slugify(input) {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'page';
}

async function ensureDir(target) {
  await fs.mkdir(target, { recursive: true });
}

async function captureInfoCard(page, screenshotPath, title, rows) {
  if (!rows || rows.length === 0) {
    return;
  }

  const cardId = `__seo_capture_${Math.random().toString(36).slice(2)}`;

  await page.evaluate(({ id, heading, lines }) => {
    const existing = document.getElementById(id);
    if (existing) {
      existing.remove();
    }

    const card = document.createElement('section');
    card.id = id;
    card.style.position = 'fixed';
    card.style.top = '24px';
    card.style.left = '24px';
    card.style.zIndex = '2147483647';
    card.style.maxWidth = '640px';
    card.style.padding = '24px';
    card.style.background = '#ffffff';
    card.style.border = '1px solid #d0d7de';
    card.style.boxShadow = '0 12px 24px rgba(15, 23, 42, 0.18)';
    card.style.fontFamily = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    card.style.lineHeight = '1.5';
    card.style.color = '#0f172a';

    const headingEl = document.createElement('h2');
    headingEl.textContent = heading;
    headingEl.style.margin = '0 0 16px 0';
    headingEl.style.fontSize = '24px';
    card.appendChild(headingEl);

    lines.forEach((line) => {
      const p = document.createElement('p');
      p.textContent = line;
      p.style.margin = '0 0 12px 0';
      p.style.fontSize = '15px';
      card.appendChild(p);
    });

    document.body.appendChild(card);
  }, { id: cardId, heading: title, lines: rows.slice(0, 10) });

  const cardHandle = await page.$(`#${cardId}`);
  if (!cardHandle) {
    return;
  }

  await cardHandle.screenshot({ path: screenshotPath });
  await page.evaluate((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.remove();
    }
  }, cardId);
}

async function collectTitleMeta(page, slug, targetDir) {
  const info = await page.evaluate(() => {
    const metaDescription = document.querySelector('meta[name="description"]');
    const metaRobots = document.querySelector('meta[name="robots"]');
    const canonical = document.querySelector('link[rel="canonical"]');
    return {
      title: document.title || '',
      description: metaDescription ? metaDescription.getAttribute('content') : '',
      robots: metaRobots ? metaRobots.getAttribute('content') : '',
      canonical: canonical ? canonical.href : '',
    };
  });

  const lines = [];
  lines.push(`Title: ${info.title || '—'}`);
  lines.push(`Meta description: ${info.description || '—'}`);
  if (info.robots) {
    lines.push(`Meta robots: ${info.robots}`);
  }
  if (info.canonical) {
    lines.push(`Canonical: ${info.canonical}`);
  }

  await captureInfoCard(page, path.join(targetDir, `${slug}-title-meta.png`), 'Title & Meta', lines);
}

async function collectHeadings(page, slug, targetDir) {
  const headings = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map((node) => ({
      tag: node.tagName,
      text: (node.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 160),
    }));
  });

  if (!headings.length) {
    return;
  }

  const lines = headings.slice(0, 12).map((heading, index) => `#${index + 1} ${heading.tag}: ${heading.text || '—'}`);
  await captureInfoCard(page, path.join(targetDir, `${slug}-headings.png`), 'Headings', lines);
}

async function collectSchema(page, slug, targetDir) {
  const schemaInfo = await page.evaluate(() => {
    const ldScripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]')).map((script) => {
      try {
        const data = JSON.parse(script.textContent || '{}');
        if (Array.isArray(data)) {
          return data.flatMap((item) => item['@type']).filter(Boolean);
        }
        return [data['@type']].flat().filter(Boolean);
      } catch (err) {
        return [];
      }
    });

    const microdataItems = Array.from(document.querySelectorAll('[itemscope][itemtype]')).map((el) => el.getAttribute('itemtype'));

    return {
      ld: ldScripts.flat().filter(Boolean),
      micro: microdataItems.filter(Boolean),
    };
  });

  const hasSchema = schemaInfo.ld.length > 0 || schemaInfo.micro.length > 0;

  const lines = [];
  lines.push(`Structured data found: ${hasSchema ? 'yes' : 'no'}`);
  if (schemaInfo.ld.length) {
    lines.push(`JSON-LD types: ${schemaInfo.ld.slice(0, 10).join(', ')}`);
  }
  if (schemaInfo.micro.length) {
    lines.push(`Microdata types: ${schemaInfo.micro.slice(0, 10).join(', ')}`);
  }

  await captureInfoCard(page, path.join(targetDir, `${slug}-schema.png`), 'Schema Markup', lines);
}

async function collectImages(page, slug, targetDir) {
  const images = await page.evaluate(() => {
    return Array.from(document.querySelectorAll('img[src]')).map((img) => ({
      src: img.currentSrc || img.src,
      alt: (img.getAttribute('alt') || '').replace(/\s+/g, ' ').trim(),
      width: img.naturalWidth,
      height: img.naturalHeight,
    }));
  });

  if (!images.length) {
    return;
  }

  const lines = images.slice(0, 10).map((img, index) => {
    const sizePart = img.width && img.height ? ` (${img.width}x${img.height})` : '';
    return `#${index + 1}: ${img.src}${sizePart}${img.alt ? ` — alt: ${img.alt}` : ''}`;
  });

  await captureInfoCard(page, path.join(targetDir, `${slug}-images.png`), 'Images', lines);
}

async function collectResponsiveness(page, slug, targetDir, maxClipHeight) {
  const desktopWidth = 1280;
  const mobileWidth = 390;
  const viewportHeight = Math.max(maxClipHeight, 800);

  await page.setViewport({ width: desktopWidth, height: viewportHeight });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(700);
  const desktopShot = await page.screenshot({
    encoding: 'base64',
    clip: { x: 0, y: 0, width: desktopWidth, height: Math.min(viewportHeight, maxClipHeight) },
  });

  await page.setViewport({ width: mobileWidth, height: viewportHeight });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(900);
  const mobileShot = await page.screenshot({
    encoding: 'base64',
    clip: { x: 0, y: 0, width: mobileWidth, height: Math.min(viewportHeight, maxClipHeight) },
  });

  await page.setViewport({ width: desktopWidth, height: viewportHeight });
  await page.evaluate(() => window.scrollTo(0, 0));

  const respInfo = await page.evaluate(() => {
    const viewportMeta = document.querySelector('meta[name="viewport"]');
    return {
      hasViewport: Boolean(viewportMeta),
      viewportContent: viewportMeta ? viewportMeta.getAttribute('content') : '',
      hasResponsiveImage: Array.from(document.querySelectorAll('img[srcset]')).length > 0,
    };
  });

  const cardId = `__seo_capture_${Math.random().toString(36).slice(2)}`;

  await page.evaluate(({ id, info, desktopBase64, mobileBase64 }) => {
    const existing = document.getElementById(id);
    if (existing) {
      existing.remove();
    }

    const card = document.createElement('section');
    card.id = id;
    card.style.position = 'fixed';
    card.style.top = '24px';
    card.style.left = '24px';
    card.style.zIndex = '2147483647';
    card.style.maxWidth = '960px';
    card.style.padding = '24px';
    card.style.background = '#ffffff';
    card.style.border = '1px solid #d0d7de';
    card.style.boxShadow = '0 12px 24px rgba(15, 23, 42, 0.18)';
    card.style.fontFamily = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    card.style.lineHeight = '1.5';
    card.style.color = '#0f172a';

    const title = document.createElement('h2');
    title.textContent = 'Responsiveness';
    title.style.margin = '0 0 16px 0';
    title.style.fontSize = '24px';
    card.appendChild(title);

    const meta = document.createElement('p');
    meta.textContent = info.hasViewport ? `Viewport meta: ${info.viewportContent}` : 'Viewport meta: missing';
    meta.style.margin = '0 0 12px 0';
    card.appendChild(meta);

    const responsiveImg = document.createElement('p');
    responsiveImg.textContent = info.hasResponsiveImage ? 'Responsive images detected (srcset present).' : 'No responsive images detected.';
    responsiveImg.style.margin = '0 0 18px 0';
    card.appendChild(responsiveImg);

    const gallery = document.createElement('div');
    gallery.style.display = 'flex';
    gallery.style.gap = '16px';
    gallery.style.alignItems = 'flex-start';

    const desktopFigure = document.createElement('figure');
    const desktopLabel = document.createElement('figcaption');
    desktopLabel.textContent = 'Desktop (1280px)';
    desktopLabel.style.margin = '8px 0 0 0';
    desktopLabel.style.fontSize = '14px';
    const desktopImage = document.createElement('img');
    desktopImage.src = `data:image/png;base64,${desktopBase64}`;
    desktopImage.style.width = '320px';
    desktopImage.style.border = '1px solid #e2e8f0';
    desktopImage.style.boxShadow = '0 6px 16px rgba(15, 23, 42, 0.12)';
    desktopImage.style.display = 'block';
    desktopFigure.appendChild(desktopImage);
    desktopFigure.appendChild(desktopLabel);

    const mobileFigure = document.createElement('figure');
    const mobileLabel = document.createElement('figcaption');
    mobileLabel.textContent = 'Mobile (390px)';
    mobileLabel.style.margin = '8px 0 0 0';
    mobileLabel.style.fontSize = '14px';
    const mobileImage = document.createElement('img');
    mobileImage.src = `data:image/png;base64,${mobileBase64}`;
    mobileImage.style.width = '200px';
    mobileImage.style.border = '1px solid #e2e8f0';
    mobileImage.style.boxShadow = '0 6px 16px rgba(15, 23, 42, 0.12)';
    mobileImage.style.display = 'block';
    mobileFigure.appendChild(mobileImage);
    mobileFigure.appendChild(mobileLabel);

    gallery.appendChild(desktopFigure);
    gallery.appendChild(mobileFigure);
    card.appendChild(gallery);

    document.body.appendChild(card);
  }, {
    id: cardId,
    info: respInfo,
    desktopBase64: desktopShot,
    mobileBase64: mobileShot,
  });

  const cardHandle = await page.$(`#${cardId}`);
  if (cardHandle) {
    await cardHandle.screenshot({ path: path.join(targetDir, `${slug}-responsiveness.png`) });
    await page.evaluate((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.remove();
      }
    }, cardId);
  }
}

async function main() {
  let parsedUrl;
  try {
    parsedUrl = new URL(urlArg);
  } catch (err) {
    console.error(`Invalid URL provided: ${urlArg}`);
    process.exit(1);
  }

  const slugParts = [parsedUrl.hostname, parsedUrl.pathname.replace(/\/+$/, '')];
  const slug = slugify(slugParts.join('-'));
  const targetDir = path.resolve(process.cwd(), outputDir);

  await ensureDir(targetDir);

  const browser = await puppeteer.launch({ headless: 'new', defaultViewport: { width: 1280, height: Math.max(clipHeight, 900) } });
  const page = await browser.newPage();

  try {
    await page.goto(parsedUrl.href, { waitUntil: 'networkidle2', timeout: 60000 });
  } catch (err) {
    console.error(`Failed to load ${parsedUrl.href}:`, err.message);
    await browser.close();
    process.exit(1);
  }

  await page.waitForTimeout(1000);
  await page.evaluate(() => window.scrollTo(0, 0));

  await collectTitleMeta(page, slug, targetDir);
  await collectHeadings(page, slug, targetDir);
  await collectSchema(page, slug, targetDir);
  await collectImages(page, slug, targetDir);
  await collectResponsiveness(page, slug, targetDir, clipHeight);

  await browser.close();

  console.log(`Screenshots saved to ${targetDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
