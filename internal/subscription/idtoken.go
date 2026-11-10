package subscription

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"net/http"
	"strings"
)

type discoveryDoc struct {
	Issuer             string `json:"issuer"`
	JWKSURI            string `json:"jwks_uri"`
	RevocationEndpoint string `json:"revocation_endpoint"`
}

func (c *Client) getJSON(ctx context.Context, rawURL string, into any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return errors.New("auth origin unreachable")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errors.New("auth origin refused the request")
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxAuthResponseBytes))
	if err != nil {
		return err
	}
	return json.Unmarshal(body, into)
}

// discovery reads the issuer's metadata. Every endpoint taken from it must
// sit on the issuer's own origin.
func (c *Client) discovery(ctx context.Context) (*discoveryDoc, error) {
	var doc discoveryDoc
	if err := c.getJSON(ctx, c.Plan.DiscoveryURL, &doc); err != nil {
		return nil, err
	}
	if doc.Issuer != c.Plan.Issuer {
		return nil, errors.New("discovery document names another issuer")
	}
	return &doc, nil
}

type jwk struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	Use string `json:"use"`
	N   string `json:"n"`
	E   string `json:"e"`
	Crv string `json:"crv"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

// IDClaims are the identity claims kept from a verified ID token.
type IDClaims struct {
	Subject string
	Email   string
}

func b64url(s string) ([]byte, error) {
	return base64.RawURLEncoding.DecodeString(strings.TrimRight(s, "="))
}

// verifyIDToken checks an ID token's signature against the issuer's published
// keys, then its issuer, audience, expiry and nonce.
func (c *Client) verifyIDToken(ctx context.Context, token, clientID, nonce string) (*IDClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errors.New("not a signed token")
	}
	headerRaw, err := b64url(parts[0])
	if err != nil {
		return nil, errors.New("malformed header")
	}
	payloadRaw, err := b64url(parts[1])
	if err != nil {
		return nil, errors.New("malformed payload")
	}
	sig, err := b64url(parts[2])
	if err != nil {
		return nil, errors.New("malformed signature")
	}
	var header struct {
		Alg string `json:"alg"`
		Kid string `json:"kid"`
	}
	if err := json.Unmarshal(headerRaw, &header); err != nil {
		return nil, errors.New("malformed header")
	}
	if header.Alg != "RS256" && header.Alg != "ES256" {
		return nil, errors.New("unsupported signing algorithm")
	}

	disc, err := c.discovery(ctx)
	if err != nil {
		return nil, err
	}
	if disc.JWKSURI == "" || !sameOrigin(disc.JWKSURI, c.Plan.Issuer) {
		return nil, errors.New("signing keys are not published on the issuer's origin")
	}
	var set struct {
		Keys []jwk `json:"keys"`
	}
	if err := c.getJSON(ctx, disc.JWKSURI, &set); err != nil {
		return nil, err
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	verified := false
	for _, k := range set.Keys {
		if header.Kid != "" && k.Kid != header.Kid {
			continue
		}
		if k.Use != "" && k.Use != "sig" {
			continue
		}
		if verifyWithKey(k, header.Alg, digest[:], sig) {
			verified = true
			break
		}
	}
	if !verified {
		return nil, errors.New("signature does not verify")
	}

	var claims struct {
		Iss   string  `json:"iss"`
		Sub   string  `json:"sub"`
		Aud   any     `json:"aud"`
		Exp   float64 `json:"exp"`
		Nonce string  `json:"nonce"`
		Email string  `json:"email"`
	}
	if err := json.Unmarshal(payloadRaw, &claims); err != nil {
		return nil, errors.New("malformed claims")
	}
	if claims.Iss != c.Plan.Issuer {
		return nil, errors.New("wrong issuer")
	}
	if !audienceContains(claims.Aud, clientID) {
		return nil, errors.New("wrong audience")
	}
	if claims.Exp == 0 || int64(claims.Exp) <= c.now().Unix() {
		return nil, errors.New("expired")
	}
	if nonce == "" || claims.Nonce != nonce {
		return nil, errors.New("wrong nonce")
	}
	if claims.Sub == "" {
		return nil, errors.New("no subject")
	}
	return &IDClaims{Subject: claims.Sub, Email: claims.Email}, nil
}

func audienceContains(aud any, want string) bool {
	switch v := aud.(type) {
	case string:
		return v == want
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok && s == want {
				return true
			}
		}
	}
	return false
}

func verifyWithKey(k jwk, alg string, digest, sig []byte) bool {
	switch {
	case alg == "RS256" && k.Kty == "RSA":
		n, err1 := b64url(k.N)
		e, err2 := b64url(k.E)
		if err1 != nil || err2 != nil || len(n) < 256 || len(e) == 0 || len(e) > 4 {
			return false
		}
		exp := 0
		for _, b := range e {
			exp = exp<<8 | int(b)
		}
		pub := &rsa.PublicKey{N: new(big.Int).SetBytes(n), E: exp}
		return rsa.VerifyPKCS1v15(pub, crypto.SHA256, digest, sig) == nil
	case alg == "ES256" && k.Kty == "EC" && k.Crv == "P-256":
		x, err1 := b64url(k.X)
		y, err2 := b64url(k.Y)
		if err1 != nil || err2 != nil || len(sig) != 64 {
			return false
		}
		pub := &ecdsa.PublicKey{Curve: elliptic.P256(), X: new(big.Int).SetBytes(x), Y: new(big.Int).SetBytes(y)}
		if !pub.Curve.IsOnCurve(pub.X, pub.Y) {
			return false
		}
		return ecdsa.Verify(pub, digest, new(big.Int).SetBytes(sig[:32]), new(big.Int).SetBytes(sig[32:]))
	}
	return false
}
