# Spring Security

> TODO: Interview-focused guide — authentication and authorization in Spring

## Topics to Cover
- **Security filter chain** — how requests flow through filters
- **Authentication** — `AuthenticationManager`, `UserDetailsService`, `PasswordEncoder`
- **Authorization** — `@PreAuthorize`, `@Secured`, role-based vs permission-based
- **JWT authentication** — token generation, validation, stateless sessions
- **OAuth2 / OpenID Connect** — authorization code flow, resource server, `@EnableOAuth2Client`
- **CORS** — configuration, `@CrossOrigin`, global CORS config
- **CSRF** — when to enable/disable, token-based protection
- **Method-level security** — `@PreAuthorize("hasRole('ADMIN')")`, SpEL expressions
- **Session management** — stateless vs stateful, concurrent session control
- **Common interview questions** — how would you secure a REST API? JWT vs session? OAuth2 flow?
