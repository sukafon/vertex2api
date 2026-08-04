package model

import "sync"

// ==================== VertexRequest Pool ====================

var vertexReqPool = sync.Pool{
	New: func() interface{} {
		return &VertexRequest{
			Variables: make(map[string]interface{}, 8),
		}
	},
}

func AcquireVertexRequest() *VertexRequest {
	req := vertexReqPool.Get().(*VertexRequest)
	req.QuerySignature = ""
	req.OperationName = ""
	if req.Variables == nil {
		req.Variables = make(map[string]interface{}, 8)
	} else {
		// 清空 map
		for k := range req.Variables {
			delete(req.Variables, k)
		}
	}
	return req
}

func ReleaseVertexRequest(req *VertexRequest) {
	if req == nil {
		return
	}
	vertexReqPool.Put(req)
}
