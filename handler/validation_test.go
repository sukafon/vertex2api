package handler

import "testing"

func TestValidateModelNameRequiresCatalogByDefault(t *testing.T) {
	if got := validateModelName("gemini-3.5-flash", false); got != "" {
		t.Fatalf("known model rejected: %s", got)
	}
	if got := validateModelName("custom-model", false); got == "" {
		t.Fatal("unknown model accepted while custom model names are disabled")
	}
	if got := validateModelName("custom-model", true); got != "" {
		t.Fatalf("custom model rejected when enabled: %s", got)
	}
}

func TestValidateModelNameRejectsPathTraversalRegardlessOfCustomModelSetting(t *testing.T) {
	for _, allowCustom := range []bool{false, true} {
		if got := validateModelName("../../custom-model", allowCustom); got == "" {
			t.Fatalf("path traversal accepted with allowCustom=%v", allowCustom)
		}
	}
}
