# Rate Limiting & Throttling

> TODO: Detailed guide with examples, diagrams, and interview questions

## Topics to Cover
- **Token bucket** algorithm — bursty traffic, refill rate
- **Leaky bucket** algorithm — smooth output rate
- **Fixed window counter** — simple but boundary issues
- **Sliding window log** — precise but memory-heavy
- **Sliding window counter** — hybrid approach
- Distributed rate limiting — Redis-based, race conditions
- Rate limiting at different layers (API gateway, application, IP-based, user-based)
- HTTP 429 and retry-after headers
- Throttling vs rate limiting vs debouncing
