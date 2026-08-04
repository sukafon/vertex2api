package stats

import (
	"fmt"
	"sort"
	"sync"
)

type ProxyStat struct {
	URL     string `json:"url,omitempty"`
	Total   int64  `json:"total"`
	Success int64  `json:"success"`
	Rate    string `json:"rate"`
}

var (
	mu    sync.RWMutex
	stats = make(map[string]*ProxyStat)
)

func RecordRequest(url string) {
	if url == "" {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	s, ok := stats[url]
	if !ok {
		s = &ProxyStat{}
		stats[url] = s
	}
	s.Total++
}

func RecordSuccess(url string) {
	if url == "" {
		return
	}
	mu.Lock()
	defer mu.Unlock()
	s, ok := stats[url]
	if !ok {
		s = &ProxyStat{}
		stats[url] = s
	}
	s.Success++
}

func GetStats() []ProxyStat {
	mu.RLock()
	defer mu.RUnlock()
	res := make([]ProxyStat, 0, len(stats))
	for k, v := range stats {
		stat := ProxyStat{
			URL:     k,
			Total:   v.Total,
			Success: v.Success,
		}
		if stat.Total > 0 {
			stat.Rate = fmt.Sprintf("%.2f%%", (float64(stat.Success)/float64(stat.Total))*100)
		} else {
			stat.Rate = "0.00%"
		}
		res = append(res, stat)
	}

	sort.Slice(res, func(i, j int) bool {
		rateI := 0.0
		if res[i].Total > 0 {
			rateI = float64(res[i].Success) / float64(res[i].Total)
		}
		rateJ := 0.0
		if res[j].Total > 0 {
			rateJ = float64(res[j].Success) / float64(res[j].Total)
		}
		if rateI == rateJ {
			return res[i].Total > res[j].Total // if rates are equal, the one with more requests goes first
		}
		return rateI > rateJ
	})

	return res
}
