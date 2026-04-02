# Threat Model

## Assets
- operator workflow documentation
- deployment and runbook instructions
- local development topology
- provider integration code
- research and review artifacts

## Trust Boundaries
- public repo readers vs internal operator environment
- public docs vs internal-only runbooks and review artifacts
- browser UI placeholders vs real runtime credentials
- public data provider integrations vs authenticated management flows

## Main Disclosure Risks
1. leaking local usernames and filesystem layout in public docs
2. leaking internal bridge topology / localhost port conventions in public-facing runbooks
3. accidentally checking in real auth material while tests and placeholders use token-shaped strings
4. overexposing provider request constants that may be mistaken for secrets
