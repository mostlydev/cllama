package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/feeds"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

const (
	defaultRecallTimeout = 300 * time.Millisecond
	defaultRetainTimeout = 2 * time.Second
	maxMemoryBlockBytes  = 16 * 1024

	memoryStatusSkipped   = "skipped"
	memoryStatusSucceeded = "succeeded"
	memoryStatusFailed    = "failed"
	memoryStatusTimedOut  = "timed_out"
)

type memoryRecallRequest struct {
	AgentID  string         `json:"agent_id"`
	Pod      string         `json:"pod,omitempty"`
	Messages any            `json:"messages,omitempty"`
	System   any            `json:"system,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type memoryRetainRequest struct {
	AgentID  string               `json:"agent_id"`
	Pod      string               `json:"pod,omitempty"`
	Metadata map[string]any       `json:"metadata,omitempty"`
	Entry    sessionhistory.Entry `json:"entry"`
}

type memoryRecallResponse struct {
	Memories []memoryBlock `json:"memories"`
}

type memoryBlock struct {
	Text   string  `json:"text"`
	Score  float64 `json:"score,omitempty"`
	Kind   string  `json:"kind,omitempty"`
	Source string  `json:"source,omitempty"`
	TS     string  `json:"ts,omitempty"`
}

func (h *Handler) recallOpenAIMemory(reqCtx context.Context, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, payload map[string]any) {
	block, err := h.recallMemory(reqCtx, agentID, agentCtx, requestedModel, memoryRecallRequest{
		AgentID:  agentID,
		Pod:      agentCtx.MetadataString("pod"),
		Messages: payload["messages"],
		Metadata: memoryMetadata(agentCtx, "openai", requestedModel),
	})
	if err != nil {
		return
	}
	if block != "" {
		feeds.InjectOpenAI(payload, block)
	}
}

func (h *Handler) recallAnthropicMemory(reqCtx context.Context, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, payload map[string]any) {
	block, err := h.recallMemory(reqCtx, agentID, agentCtx, requestedModel, memoryRecallRequest{
		AgentID:  agentID,
		Pod:      agentCtx.MetadataString("pod"),
		Messages: payload["messages"],
		System:   payload["system"],
		Metadata: memoryMetadata(agentCtx, "anthropic", requestedModel),
	})
	if err != nil {
		return
	}
	if block != "" {
		feeds.InjectAnthropic(payload, block)
	}
}

func (h *Handler) recallMemory(reqCtx context.Context, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, payload memoryRecallRequest) (string, error) {
	if agentCtx == nil || agentCtx.Memory == nil || agentCtx.Memory.Recall == nil {
		if agentCtx != nil && agentCtx.Memory != nil {
			h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
				Service:   agentCtx.Memory.Service,
				Operation: "recall",
				Status:    memoryStatusSkipped,
				LatencyMS: 0,
			})
		}
		return "", nil
	}

	timeout := defaultRecallTimeout
	if agentCtx.Memory.Recall.TimeoutMS > 0 {
		timeout = time.Duration(agentCtx.Memory.Recall.TimeoutMS) * time.Millisecond
	}
	start := time.Now()
	ctx, cancel := context.WithTimeout(reqCtx, timeout)
	defer cancel()

	resp, err := doMemoryJSONRequest(ctx, h.client, http.MethodPost, agentCtx.Memory.BaseURL+agentCtx.Memory.Recall.Path, agentCtx.Memory.Auth, payload)
	if err != nil {
		h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
			Service:   agentCtx.Memory.Service,
			Operation: "recall",
			Status:    memoryErrorStatus(err),
			LatencyMS: time.Since(start).Milliseconds(),
			Error:     err,
		})
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		err := fmt.Errorf("recall returned status %d", resp.StatusCode)
		h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
			Service:    agentCtx.Memory.Service,
			Operation:  "recall",
			Status:     memoryStatusFailed,
			StatusCode: resp.StatusCode,
			LatencyMS:  time.Since(start).Milliseconds(),
			Error:      err,
		})
		return "", err
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxMemoryBlockBytes+1))
	if err != nil {
		h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
			Service:    agentCtx.Memory.Service,
			Operation:  "recall",
			Status:     memoryStatusFailed,
			StatusCode: resp.StatusCode,
			LatencyMS:  time.Since(start).Milliseconds(),
			Error:      err,
		})
		return "", err
	}
	if len(body) > maxMemoryBlockBytes {
		err := fmt.Errorf("recall response exceeds %d bytes", maxMemoryBlockBytes)
		h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
			Service:    agentCtx.Memory.Service,
			Operation:  "recall",
			Status:     memoryStatusFailed,
			StatusCode: resp.StatusCode,
			LatencyMS:  time.Since(start).Milliseconds(),
			Error:      err,
		})
		return "", err
	}

	var decoded memoryRecallResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		err := fmt.Errorf("parse recall response: %w", err)
		h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
			Service:    agentCtx.Memory.Service,
			Operation:  "recall",
			Status:     memoryStatusFailed,
			StatusCode: resp.StatusCode,
			LatencyMS:  time.Since(start).Milliseconds(),
			Error:      err,
		})
		return "", err
	}

	block := formatMemoryBlocks(agentCtx.Memory.Service, decoded.Memories)
	blocks := len(decoded.Memories)
	injectedBytes := len(block)
	h.logger.LogMemoryOp(agentID, requestedModel, logging.MemoryOpInfo{
		Service:       agentCtx.Memory.Service,
		Operation:     "recall",
		Status:        memoryStatusSucceeded,
		StatusCode:    resp.StatusCode,
		LatencyMS:     time.Since(start).Milliseconds(),
		Blocks:        &blocks,
		InjectedBytes: &injectedBytes,
	})
	return block, nil
}

func (h *Handler) retainMemory(agentID string, agentCtx *agentctx.AgentContext, entry sessionhistory.Entry) {
	if agentCtx == nil || agentCtx.Memory == nil || agentCtx.Memory.Retain == nil {
		if agentCtx != nil && agentCtx.Memory != nil {
			h.logger.LogMemoryOp(agentID, entry.RequestedModel, logging.MemoryOpInfo{
				Service:   agentCtx.Memory.Service,
				Operation: "retain",
				Status:    memoryStatusSkipped,
				LatencyMS: 0,
			})
		}
		return
	}

	go func() {
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), defaultRetainTimeout)
		defer cancel()
		resp, err := doMemoryJSONRequest(ctx, h.client, http.MethodPost, agentCtx.Memory.BaseURL+agentCtx.Memory.Retain.Path, agentCtx.Memory.Auth, memoryRetainRequest{
			AgentID:  agentID,
			Pod:      agentCtx.MetadataString("pod"),
			Metadata: memoryMetadata(agentCtx, "retain", entry.RequestedModel),
			Entry:    entry,
		})
		if err == nil && resp != nil {
			defer resp.Body.Close()
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				err = fmt.Errorf("retain returned status %d", resp.StatusCode)
			}
		}
		latency := time.Since(start).Milliseconds()
		if err != nil {
			statusCode := 0
			if resp != nil {
				statusCode = resp.StatusCode
			}
			h.logger.LogMemoryOp(agentID, entry.RequestedModel, logging.MemoryOpInfo{
				Service:    agentCtx.Memory.Service,
				Operation:  "retain",
				Status:     memoryErrorStatus(err),
				StatusCode: statusCode,
				LatencyMS:  latency,
				Error:      err,
			})
			return
		}
		h.logger.LogMemoryOp(agentID, entry.RequestedModel, logging.MemoryOpInfo{
			Service:    agentCtx.Memory.Service,
			Operation:  "retain",
			Status:     memoryStatusSucceeded,
			StatusCode: resp.StatusCode,
			LatencyMS:  latency,
		})
	}()
}

func doMemoryJSONRequest(ctx context.Context, client *http.Client, method, url string, auth *agentctx.AuthEntry, payload any) (*http.Response, error) {
	if client == nil {
		client = &http.Client{}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if auth != nil && strings.EqualFold(auth.Type, "bearer") && auth.Token != "" {
		req.Header.Set("Authorization", "Bearer "+auth.Token)
	}
	return client.Do(req)
}

func formatMemoryBlocks(service string, memories []memoryBlock) string {
	if len(memories) == 0 {
		return ""
	}

	var b strings.Builder
	fmt.Fprintf(&b, "--- BEGIN MEMORY (from %s) ---\n", service)
	for _, m := range memories {
		text := strings.TrimSpace(m.Text)
		if text == "" {
			continue
		}
		prefixParts := make([]string, 0, 4)
		if m.Kind != "" {
			prefixParts = append(prefixParts, "kind="+m.Kind)
		}
		if m.Source != "" {
			prefixParts = append(prefixParts, "source="+m.Source)
		}
		if m.TS != "" {
			prefixParts = append(prefixParts, "ts="+m.TS)
		}
		if m.Score != 0 {
			prefixParts = append(prefixParts, fmt.Sprintf("score=%.2f", m.Score))
		}
		if len(prefixParts) > 0 {
			fmt.Fprintf(&b, "[%s]\n", strings.Join(prefixParts, ", "))
		}
		b.WriteString(text)
		if !strings.HasSuffix(text, "\n") {
			b.WriteByte('\n')
		}
	}
	fmt.Fprintf(&b, "--- END MEMORY ---")
	return b.String()
}

func memoryMetadata(agentCtx *agentctx.AgentContext, path, requestedModel string) map[string]any {
	if agentCtx == nil {
		return nil
	}
	meta := map[string]any{
		"service": agentCtx.MetadataString("service"),
		"type":    agentCtx.MetadataString("type"),
		"path":    path,
	}
	if timezone := agentCtx.MetadataString("timezone"); timezone != "" {
		meta["timezone"] = timezone
	}
	if requestedModel != "" {
		meta["requested_model"] = requestedModel
	}
	return meta
}

func memoryErrorStatus(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return memoryStatusTimedOut
	}
	return memoryStatusFailed
}
