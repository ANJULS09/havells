FROM node:22-alpine

WORKDIR /app

# Install dependencies first for layer caching
COPY package.json package-lock.json* ./
RUN npm install

# Copy application files
COPY . .

# Expose Next.js default port
EXPOSE 3000

# Run Next.js dev server
CMD ["npm", "run", "dev"]
