import { useEffect, useRef, useState, useCallback } from 'react';

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: ArrayBuffer) => void;
  onError?: (error: Event) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
}

export interface UseWebSocketReturn {
  status: WebSocketStatus;
  send: (data: string | ArrayBuffer | Blob) => void;
  close: () => void;
  reconnect: () => void;
}

/**
 * WebSocket hook for binary data streaming
 * Handles connection management, reconnection, and binary message parsing
 */
export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    onMessage,
    onError,
    reconnect = true,
    reconnectInterval = 3000,
    reconnectAttempts = 10,
  } = options;

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const attemptCountRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  
  // Stable callback refs to prevent reconnection on parent re-renders
  const onMessageRef = useRef(onMessage);
  const onErrorRef = useRef(onError);
  
  // Update refs when callbacks change (no reconnection triggered)
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);
  
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setStatus('connecting');

    try {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer'; // Critical for binary data

      ws.onopen = (_event) => {
        console.log('[WebSocket] Connected to', url);
        setStatus('connected');
        attemptCountRef.current = 0;
      };

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          onMessageRef.current?.(event.data);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setStatus('error');
        onErrorRef.current?.(error);
      };

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected');
        setStatus('disconnected');
        wsRef.current = null;

        // Attempt reconnection
        if (
          reconnect &&
          shouldReconnectRef.current &&
          attemptCountRef.current < reconnectAttempts
        ) {
          attemptCountRef.current++;
          console.log(
            `[WebSocket] Reconnecting... (${attemptCountRef.current}/${reconnectAttempts})`
          );
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error);
      setStatus('error');
    }
  }, [url, reconnect, reconnectInterval, reconnectAttempts]); // Removed onMessage, onError - using refs

  const send = useCallback((data: string | ArrayBuffer | Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    } else {
      console.warn('[WebSocket] Cannot send, connection not open');
    }
  }, []);

  const close = useCallback(() => {
    shouldReconnectRef.current = false;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('disconnected');
  }, []);

  const manualReconnect = useCallback(() => {
    attemptCountRef.current = 0;
    shouldReconnectRef.current = true;
    connect();
  }, [connect]);

  useEffect(() => {
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  return {
    status,
    send,
    close,
    reconnect: manualReconnect,
  };
}
