# SmartAutoDJ website

Static project website for demo and deployment.

## Preview locally

Open `website/index.html` in a browser. No build step or backend is required.

## Add demo audio

1. Put exported `.wav` or `.mp3` files in `website/audio/`.
2. Open `website/app.js`.
3. Set each `demoTracks` entry's `file` value, for example:

```js
file: "audio/song-a__song-b__tier2.wav"
```

## Deploy

Deploy the `website/` folder as a static site on GitHub Pages, Netlify, Vercel,
or any static file host.
