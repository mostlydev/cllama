package identity

import "testing"

func TestParseBearerValid(t *testing.T) {
	id, secret, err := ParseBearer("Bearer tiverton:abc123def456")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if id != "tiverton" {
		t.Errorf("expected id 'tiverton', got %q", id)
	}
	if secret != "abc123def456" {
		t.Errorf("expected secret 'abc123def456', got %q", secret)
	}
}

func TestParseBearerColonInSecret(t *testing.T) {
	id, secret, err := ParseBearer("Bearer bot-a:secret:with:colons")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if id != "bot-a" {
		t.Errorf("expected id 'bot-a', got %q", id)
	}
	if secret != "secret:with:colons" {
		t.Errorf("expected secret 'secret:with:colons', got %q", secret)
	}
}

func TestParseBearerMissingColon(t *testing.T) {
	_, _, err := ParseBearer("Bearer noseparator")
	if err == nil {
		t.Error("expected error for missing colon")
	}
}

func TestParseBearerEmpty(t *testing.T) {
	_, _, err := ParseBearer("")
	if err == nil {
		t.Error("expected error for empty header")
	}
}

func TestParseBearerNoPrefix(t *testing.T) {
	_, _, err := ParseBearer("Basic tiverton:abc")
	if err == nil {
		t.Error("expected error for non-Bearer auth")
	}
}
