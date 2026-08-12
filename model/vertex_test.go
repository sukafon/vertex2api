package model

import "testing"

func TestNormalizeGroundingMetadataDropsDefaultInitializedValues(t *testing.T) {
	metadata := &GroundingMetadata{
		WebSearchQueries:   []string{"", "   "},
		SearchEntryPoint:   &SearchEntryPoint{},
		GroundingChunks:    []GroundingChunk{{Web: &WebChunk{}, RetrievedContext: map[string]interface{}{}, Maps: map[string]interface{}{}}},
		GroundingSupports:  []GroundingSupport{{Segment: &Segment{}}},
		RetrievalQueries:   []string{""},
		SourceFlaggingURIs: []map[string]interface{}{{}},
		RetrievalMetadata:  map[string]interface{}{"nested": map[string]interface{}{}, "items": []interface{}{}},
	}
	if got := NormalizeGroundingMetadata(metadata); got != nil {
		t.Fatalf("empty grounding metadata was preserved: %#v", got)
	}
}

func TestNormalizeGroundingMetadataPreservesTypedZeroAndFalse(t *testing.T) {
	metadata := &GroundingMetadata{
		RetrievalMetadata: map[string]interface{}{
			"score":   float64(0),
			"enabled": false,
		},
	}
	got := NormalizeGroundingMetadata(metadata)
	if got == nil || got.RetrievalMetadata["score"] != float64(0) || got.RetrievalMetadata["enabled"] != false {
		t.Fatalf("meaningful scalar metadata was dropped: %#v", got)
	}
}

func TestNormalizeCitationMetadataRequiresActualCitation(t *testing.T) {
	tests := []map[string]interface{}{
		{},
		{"citations": []interface{}{}},
		{"citations": []interface{}{map[string]interface{}{}}},
		{"citations": []interface{}{map[string]interface{}{"startIndex": float64(0), "endIndex": float64(0), "uri": ""}}},
	}
	for _, metadata := range tests {
		if got := NormalizeCitationMetadata(metadata); got != nil {
			t.Fatalf("empty citation metadata was preserved: %#v", got)
		}
	}

	got := NormalizeCitationMetadata(map[string]interface{}{
		"citations": []interface{}{map[string]interface{}{
			"startIndex": float64(0), "endIndex": float64(4),
			"uri": "https://example.com", "title": "Vertex-only title",
		}},
	})
	citations := got["citationSources"].([]interface{})
	if len(citations) != 1 {
		t.Fatalf("valid citation was dropped: %#v", got)
	}
	citation := citations[0].(map[string]interface{})
	if citation["uri"] != "https://example.com" || citation["endIndex"] != float64(4) {
		t.Fatalf("citation source was not translated: %#v", citation)
	}
	if _, ok := citation["title"]; ok {
		t.Fatalf("Vertex-only citation field leaked: %#v", citation)
	}
}
