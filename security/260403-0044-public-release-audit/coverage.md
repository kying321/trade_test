# Coverage

## STRIDE Covered
- Information Disclosure: covered
- Tampering: light coverage via placeholder/config review only
- Spoofing: light coverage via auth-header placeholder review only
- Denial of Service: not a focus in this pass
- Repudiation: not a focus in this pass
- Elevation of Privilege: not a focus in this pass

## OWASP Coverage
- A01 Broken Access Control: light
- A02 Cryptographic Failures: light
- A05 Security Misconfiguration: strong
- A07 Identification and Authentication Failures: light
- Others: not primary in this pass

## Adversarial Lenses Exercised
- public repo reader
- internal operator with accidental commit risk
- packaging/release reviewer
