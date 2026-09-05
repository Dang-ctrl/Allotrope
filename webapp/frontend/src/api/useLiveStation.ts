import { useEffect, useReducer, useRef } from 'react'
import type { LiveMessage, Observation, SafetyEvent, Telemetry } from '../types'

const MAX_SAFETY_EVENTS = 20
// Stations publish telemetry roughly every 2s; a socket can stay open while
// the backend's own MQTT/gRPC links are down, so "connected" is judged by
// data freshness, not merely by the WebSocket's open/closed state.
const STALE_AFTER_MS = 8_000

interface LiveState {
  telemetry: Telemetry | null
  observation: Observation | null
  safetyEvents: SafetyEvent[]
  connected: boolean
  lastMessageAt: number | null
}

type Action =
  | { kind: 'socketOpen' }
  | { kind: 'socketClosed' }
  | { kind: 'message'; message: LiveMessage }
  | { kind: 'checkStale' }

const initialState: LiveState = {
  telemetry: null,
  observation: null,
  safetyEvents: [],
  connected: false,
  lastMessageAt: null,
}

function reducer(state: LiveState, action: Action): LiveState {
  switch (action.kind) {
    case 'socketOpen':
      return state
    case 'socketClosed':
      return { ...state, connected: false }
    case 'checkStale': {
      if (!state.lastMessageAt) return state
      const stale = Date.now() - state.lastMessageAt > STALE_AFTER_MS
      return stale ? { ...state, connected: false } : state
    }
    case 'message': {
      const message = action.message
      if (message.type === 'snapshot') {
        return {
          ...state,
          telemetry: message.telemetry ?? state.telemetry,
          observation: message.observation ?? state.observation,
          safetyEvents: message.safety_events.slice(-MAX_SAFETY_EVENTS),
          connected: true,
          lastMessageAt: Date.now(),
        }
      }
      if (message.type === 'telemetry') {
        return { ...state, telemetry: message.data, connected: true, lastMessageAt: Date.now() }
      }
      if (message.type === 'observation') {
        return { ...state, observation: message.data }
      }
      if (message.type === 'safety') {
        const event: SafetyEvent = { ...message.data, ts: message.ts }
        return { ...state, safetyEvents: [...state.safetyEvents, event].slice(-MAX_SAFETY_EVENTS) }
      }
      return state
    }
  }
}

/** One WebSocket per station, shared by every widget on that station's dashboard. */
export function useLiveStation(stationId: string): LiveState {
  const [state, dispatch] = useReducer(reducer, initialState)
  const reconnectAttempt = useRef(0)

  useEffect(() => {
    let cancelled = false
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      if (cancelled) return
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/stations/${stationId}`)

      socket.onopen = () => {
        reconnectAttempt.current = 0
        dispatch({ kind: 'socketOpen' })
      }
      socket.onmessage = (event) => {
        dispatch({ kind: 'message', message: JSON.parse(event.data) })
      }
      socket.onclose = () => {
        dispatch({ kind: 'socketClosed' })
        if (cancelled) return
        const delay = Math.min(1000 * 2 ** reconnectAttempt.current, 10000)
        reconnectAttempt.current += 1
        reconnectTimer = setTimeout(connect, delay)
      }
      socket.onerror = () => {
        socket?.close()
      }
    }

    connect()
    const staleCheck = setInterval(() => dispatch({ kind: 'checkStale' }), 2000)

    return () => {
      cancelled = true
      clearInterval(staleCheck)
      if (reconnectTimer) clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [stationId])

  return state
}
