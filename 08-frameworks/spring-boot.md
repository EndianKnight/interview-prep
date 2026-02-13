# Spring Boot

> TODO: Interview-focused guide — not a tutorial, just what you'd discuss in an interview

## Topics to Cover
- Spring Boot vs Spring Framework — what auto-configuration solves
- **Key annotations** — `@SpringBootApplication`, `@Component`, `@Service`, `@Repository`, `@Controller`, `@RestController`
- **Dependency injection** — `@Autowired`, `@Qualifier`, `@Primary`, constructor injection (preferred)
- **Bean lifecycle** — `@Bean`, `@Configuration`, `@PostConstruct`, `@PreDestroy`, scopes (singleton, prototype, request, session)
- **Application properties** — `application.yml`, `@Value`, `@ConfigurationProperties`, profiles (`@Profile`, `spring.profiles.active`)
- **REST APIs** — `@GetMapping`, `@PostMapping`, `@RequestBody`, `@PathVariable`, `@RequestParam`, `@ResponseStatus`
- **Exception handling** — `@ControllerAdvice`, `@ExceptionHandler`, custom error responses
- **Actuator** — health checks, metrics, custom endpoints
- **Auto-configuration** — how it works, `@Conditional` annotations, starter dependencies
- **Testing** — `@SpringBootTest`, `@WebMvcTest`, `@DataJpaTest`, `@MockBean`, TestRestTemplate
