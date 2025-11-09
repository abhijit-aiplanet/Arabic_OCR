# AIN OCR Frontend

Modern Next.js frontend for the AIN OCR application with a beautiful and responsive UI.

## Features

- 🎨 Modern, beautiful UI with Tailwind CSS
- 📱 Fully responsive design
- 🖼️ Drag & drop image upload
- ⚙️ Advanced settings panel
- 📋 One-click copy to clipboard
- 🌍 Support for multiple languages including Arabic (RTL)
- 🔄 Real-time processing status
- 🎯 Toast notifications for better UX

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **File Upload**: React Dropzone
- **Notifications**: React Hot Toast

## Setup

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

1. Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

2. Create environment file:

```bash
cp env.local.example .env.local
```

3. Configure environment variables in `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, use your actual backend URL:
```env
NEXT_PUBLIC_API_URL=https://your-backend-url.vercel.app
```

### Development

Run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

Build for production:

```bash
npm run build
npm start
```

## Deployment

### Deploy to Vercel (Recommended)

Vercel is the easiest way to deploy Next.js applications.

#### Option 1: Deploy via Vercel CLI

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
vercel
```

4. For production deployment:
```bash
vercel --prod
```

#### Option 2: Deploy via Vercel Dashboard

1. Push your code to GitHub
2. Visit [vercel.com](https://vercel.com)
3. Click "New Project"
4. Import your repository
5. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

6. Add Environment Variable:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: `https://your-backend-url.vercel.app`

7. Click "Deploy"

#### Option 3: Deploy via GitHub Integration

1. Connect your GitHub account to Vercel
2. Select repository
3. Vercel will auto-detect Next.js
4. Set environment variables
5. Deploy automatically on every push to main branch

### Deploy to Other Platforms

#### Netlify

1. Build command: `npm run build`
2. Publish directory: `.next`
3. Add environment variable: `NEXT_PUBLIC_API_URL`

#### Cloudflare Pages

1. Build command: `npx @cloudflare/next-on-pages@1`
2. Build output directory: `.vercel/output/static`
3. Add environment variable: `NEXT_PUBLIC_API_URL`

## Project Structure

```
frontend/
├── public/              # Static files
├── src/
│   ├── app/            # Next.js app router
│   │   ├── layout.tsx  # Root layout
│   │   └── page.tsx    # Home page
│   ├── components/     # React components
│   │   ├── ImageUploader.tsx
│   │   ├── ExtractedText.tsx
│   │   └── AdvancedSettings.tsx
│   ├── lib/           # Utilities
│   │   └── api.ts     # API client
│   └── styles/        # Global styles
│       └── globals.css
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── next.config.js
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

**Important**: All environment variables that need to be accessible in the browser must be prefixed with `NEXT_PUBLIC_`.

## API Integration

The frontend communicates with the backend via REST API:

- **POST /api/ocr**: Process OCR request
- **GET /api/prompt**: Get default prompt
- **GET /health**: Health check

See `src/lib/api.ts` for implementation details.

## Customization

### Styling

- Modify `tailwind.config.js` for theme customization
- Edit `src/styles/globals.css` for global styles
- Component styles are inline with Tailwind classes

### Components

All components are in `src/components/`:
- `ImageUploader.tsx`: Image upload with drag & drop
- `ExtractedText.tsx`: Display extracted text with copy functionality
- `AdvancedSettings.tsx`: Collapsible settings panel

### API Client

Modify `src/lib/api.ts` to change API endpoints or add new functions.

## Troubleshooting

### CORS Errors

Make sure your backend has the frontend URL in its CORS allowed origins:

```python
# backend/main.py
allow_origins=[
    "http://localhost:3000",
    "https://your-frontend.vercel.app"
]
```

### Environment Variables Not Working

1. Restart the development server after changing `.env.local`
2. Make sure variables are prefixed with `NEXT_PUBLIC_`
3. Don't commit `.env.local` to git

### Build Errors

1. Clear cache: `rm -rf .next`
2. Reinstall dependencies: `rm -rf node_modules && npm install`
3. Check Node.js version: `node --version` (should be 18+)

## Performance

The frontend is optimized for:
- Fast page loads with Next.js SSR
- Optimized images with Next.js Image component
- Code splitting for smaller bundles
- Efficient re-renders with React hooks

## Browser Support

- Chrome/Edge: Last 2 versions
- Firefox: Last 2 versions
- Safari: Last 2 versions
- Mobile browsers: iOS Safari, Chrome Android

## License

MIT

