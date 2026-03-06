import { readFile } from "node:fs/promises";

async function loadGoogleFont(font: string, text: string, weight: number): Promise<ArrayBuffer> {
  const API = `https://fonts.googleapis.com/css2?family=${font}:wght@${weight}&text=${encodeURIComponent(text)}`;

  const css = await (
    await fetch(API, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1",
      },
    })
  ).text();

  const resource = css.match(
    /src: url\((.+?)\) format\('(opentype|truetype)'\)/
  );

  if (!resource) throw new Error("Failed to download dynamic font");

  const res = await fetch(resource[1]);

  if (!res.ok) {
    throw new Error("Failed to download dynamic font. Status: " + res.status);
  }

  return res.arrayBuffer();
}

async function loadLocalFont(paths: string[]): Promise<ArrayBuffer | null> {
  for (const path of paths) {
    try {
      const data = await readFile(path);
      return Uint8Array.from(data).buffer;
    } catch {
      // Try the next candidate font path.
    }
  }

  return null;
}

async function loadGoogleFonts(
  text: string
): Promise<
  Array<{ name: string; data: ArrayBuffer; weight: number; style: string }>
> {
  const fontsConfig = [
    {
      name: "DejaVu Sans",
      font: "IBM+Plex+Mono",
      weight: 400,
      style: "normal",
      localPaths: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
      ],
    },
    {
      name: "DejaVu Sans",
      font: "IBM+Plex+Mono",
      weight: 700,
      style: "bold",
      localPaths: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
      ],
    },
  ];

  const fonts = await Promise.all(
    fontsConfig.map(async ({ name, font, weight, style, localPaths }) => {
      const localData = await loadLocalFont(localPaths);
      const data = localData ?? (await loadGoogleFont(font, text, weight));
      return { name, data, weight, style };
    })
  );

  return fonts;
}

export default loadGoogleFonts;
