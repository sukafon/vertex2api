package handler

import (
	"strings"

	"vertex2api/model"
)

func validateModelName(name string, allowCustom bool) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "model is required"
	}
	if strings.ContainsAny(name, `/\\`) || strings.Contains(name, "..") {
		return "model name contains an invalid path sequence"
	}
	if !allowCustom && !model.IsKnownModel(name) {
		return "custom model names are disabled"
	}
	return ""
}
