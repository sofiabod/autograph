"use client"

import { useEffect, useState, useCallback, useRef, useMemo } from "react"
import dynamic from "next/dynamic"

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
})

interface GraphNode {
  id: string
  type: string
  experiment_id?: number
  commit?: string
  val_bpb?: number
  delta_bpb?: number
  status?: string
  category?: string
  change_summary?: string
  reasoning?: string
  hypothesis?: string
  name?: string
  text?: string
  kept?: boolean
  run_decision?: boolean
  confidence?: number
  x?: number
  y?: number
}

interface GraphLink {
  source: string | GraphNode
  target: string | GraphNode
  edge_type: string
  delta_val_bpb?: number
  rationale?: string
  next_idea?: string
  reason?: string
  explanation?: string
  why_it_worked?: string
  why_it_failed?: string
  lesson_learned?: string
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

type Selection =
  | { kind: "node"; data: GraphNode }
  | { kind: "link"; data: GraphLink }
  | null

// polls memgraph api every N seconds for real-time updates
const API_URL = "http://127.0.0.1:8000"
const POLL_INTERVAL = 5000

function MetaRow({ label, value }: { label: string; value: string | number | boolean | undefined }) {
  if (value === undefined || value === "") return null
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ color: "#999", fontSize: 10, letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ color: "#333", fontSize: 12, marginTop: 2, lineHeight: 1.5 }}>
        {typeof value === "number" ? value.toFixed(6) : typeof value === "boolean" ? (value ? "yes" : "no") : value}
      </div>
    </div>
  )
}

function NodePanel({ node }: { node: GraphNode }) {
  if (node.type === "technique") {
    return (
      <>
        <div style={{ color: "#ccc", fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
          {node.name}
        </div>
        <MetaRow label="type" value="technique" />
        <MetaRow label="category" value={node.category} />
      </>
    )
  }

  if (node.type === "hypothesis") {
    return (
      <>
        <div style={{ color: "#e6a23c", fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
          hypothesis
        </div>
        <MetaRow label="text" value={node.text} />
        <MetaRow label="status" value={node.status} />
      </>
    )
  }

  if (node.type === "result") {
    return (
      <>
        <div style={{ color: node.kept ? "#67c23a" : "#999", fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
          result
        </div>
        <MetaRow label="text" value={node.text} />
        <MetaRow label="val_bpb" value={node.val_bpb} />
        <MetaRow label="kept" value={node.kept} />
      </>
    )
  }

  if (node.type === "debate") {
    return (
      <>
        <div style={{ color: "#9b59b6", fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
          debate
        </div>
        <MetaRow label="decision" value={node.run_decision ? "run" : "skip"} />
        <MetaRow label="change" value={node.change_summary} />
        <MetaRow label="confidence" value={node.confidence} />
        <MetaRow label="reasoning" value={node.reasoning} />
      </>
    )
  }

  if (node.type === "agent") {
    return (
      <>
        <div style={{ color: "#e74c3c", fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
          agent
        </div>
        <MetaRow label="name" value={node.name} />
      </>
    )
  }

  // experiment (default)
  return (
    <>
      <div style={{
        fontSize: 13,
        fontWeight: 600,
        marginBottom: 12,
        color: node.status === "keep" ? "#5b8def" : "#999",
      }}>
        #{node.experiment_id} — {node.change_summary}
      </div>
      <MetaRow label="status" value={node.status} />
      <MetaRow label="commit" value={node.commit} />
      <MetaRow label="val_bpb" value={node.val_bpb} />
      <MetaRow label="delta_bpb" value={node.delta_bpb} />
      <MetaRow label="category" value={node.category} />
      <MetaRow label="hypothesis" value={node.hypothesis} />
      <MetaRow label="reasoning" value={node.reasoning} />
    </>
  )
}

function LinkPanel({ link }: { link: GraphLink }) {
  const srcNode = typeof link.source === "string" ? null : link.source
  const tgtNode = typeof link.target === "string" ? null : link.target

  const getLabel = (n: GraphNode | null) => {
    if (!n) return "?"
    if (n.type === "technique") return n.name
    if (n.type === "hypothesis") return "hypothesis"
    if (n.type === "result") return "result"
    if (n.type === "debate") return "debate"
    if (n.type === "agent") return n.name
    return `#${n.experiment_id}`
  }

  return (
    <>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "#5b8def" }}>
        {getLabel(srcNode)} → {getLabel(tgtNode)}
      </div>
      <MetaRow label="edge type" value={link.edge_type} />
      <MetaRow label="delta_val_bpb" value={link.delta_val_bpb} />
      <MetaRow label="rationale" value={link.rationale} />
      <MetaRow label="next idea" value={link.next_idea} />
      <MetaRow label="reason" value={link.reason} />
      <MetaRow label="explanation" value={link.explanation} />
      <MetaRow label="why it worked" value={link.why_it_worked} />
      <MetaRow label="why it failed" value={link.why_it_failed} />
      <MetaRow label="lesson learned" value={link.lesson_learned} />
    </>
  )
}

export default function Home() {
  const [data, setData] = useState<GraphData | null>(null)
  const [selection, setSelection] = useState<Selection>(null)
  const [timeMax, setTimeMax] = useState<number>(1000)
  const graphRef = useRef<any>(null)

  const prevCountRef = useRef<number>(0)

  // poll for updates, only re-render if node count changed
  const fetchGraph = useCallback(() => {
    fetch(`${API_URL}/graph`)
      .then((r) => r.json())
      .then((d) => {
        const newCount = d.nodes.length + d.links.length
        if (newCount !== prevCountRef.current) {
          prevCountRef.current = newCount
          setData(d)
          const maxId = Math.max(0, ...d.nodes
            .filter((n: GraphNode) => n.type === "experiment")
            .map((n: GraphNode) => n.experiment_id ?? 0))
          setTimeMax(maxId)
        }
      })
      .catch(console.error)
  }, [])

  useEffect(() => {
    fetchGraph()
    const interval = setInterval(fetchGraph, POLL_INTERVAL)
    return () => clearInterval(interval)
  }, [fetchGraph])

  const [timeSlider, setTimeSlider] = useState<number>(1000)

  useEffect(() => {
    if (timeMax > 0) setTimeSlider(timeMax)
  }, [timeMax])

  const filteredData = useMemo(() => {
    if (!data) return { nodes: [], links: [] }

    const visibleNodes = data.nodes.filter((n) => {
      // non-experiment nodes always visible
      if (n.type !== "experiment") return true
      return (n.experiment_id ?? 0) <= timeSlider
    })

    const ids = new Set(visibleNodes.map((n) => n.id))
    const visibleLinks = data.links.filter((l) => {
      const src = typeof l.source === "string" ? l.source : l.source.id
      const tgt = typeof l.target === "string" ? l.target : l.target.id
      return ids.has(src) && ids.has(tgt)
    })

    return { nodes: visibleNodes, links: visibleLinks }
  }, [data, timeSlider])

  const nodeColor = useCallback((node: GraphNode) => {
    if (node.type === "technique") return "#ccc"
    if (node.type === "hypothesis") return node.status === "confirmed" ? "#67c23a" : node.status === "rejected" ? "#e74c3c" : "#e6a23c"
    if (node.type === "result") return node.kept ? "#67c23a" : "#999"
    if (node.type === "debate") return "#9b59b6"
    if (node.type === "agent") return "#e74c3c"
    if (node.status === "keep") return "#5b8def"
    return "#bbb"
  }, [])

  const nodeSize = useCallback((node: GraphNode) => {
    if (node.type === "technique") return 2
    if (node.type === "hypothesis") return 4
    if (node.type === "result") return 2
    if (node.type === "debate") return 5
    if (node.type === "agent") return 4
    if (node.status === "keep") return 6
    return 3
  }, [])

  const nodeLabel = useCallback((node: GraphNode) => {
    if (node.type === "technique") return node.name || ""
    if (node.type === "hypothesis") return `hypothesis: ${(node.text || "").slice(0, 60)}...`
    if (node.type === "result") return (node.text || "").slice(0, 60)
    if (node.type === "debate") return `debate: ${node.change_summary || ""}`
    if (node.type === "agent") return node.name || ""
    return `#${node.experiment_id} ${node.change_summary || ""}`
  }, [])

  const linkColor = useCallback((link: GraphLink) => {
    if (link.edge_type === "IMPROVED_FROM") return "rgba(91, 141, 239, 0.6)"
    if (link.edge_type === "FAILED_FROM") return "rgba(180, 180, 180, 0.3)"
    if (link.edge_type === "LED_TO") return "rgba(91, 141, 239, 0.3)"
    if (link.edge_type === "CHALLENGED") return "rgba(231, 76, 60, 0.5)"
    if (link.edge_type === "REFINES") return "rgba(230, 162, 60, 0.5)"
    if (link.edge_type === "CONTRADICTS") return "rgba(231, 76, 60, 0.6)"
    if (link.edge_type === "PRODUCED") return "rgba(103, 194, 58, 0.3)"
    return "rgba(200, 200, 200, 0.15)"
  }, [])

  const linkWidth = useCallback((link: GraphLink) => {
    if (link.edge_type === "TRIED") return 0.2
    if (link.edge_type === "LED_TO") return 0.6
    if (link.edge_type === "IMPROVED_FROM") return 1.5
    if (link.edge_type === "CHALLENGED") return 1.0
    if (link.edge_type === "CONTRADICTS") return 1.2
    if (link.edge_type === "REFINES") return 0.8
    return 0.5
  }, [])

  if (!data) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        color: "#999",
        fontSize: 14,
      }}>
        loading graph...
      </div>
    )
  }

  const panelStyle: React.CSSProperties = {
    width: 200,
    maxHeight: "50vh",
    alignSelf: "flex-start",
    background: "#fafafa",
    border: "1px solid #d4d4d4",
    borderRadius: 6,
    padding: 10,
    overflowY: "auto",
    flexShrink: 0,
    fontSize: 10,
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", width: "100vw", height: "100vh", background: "#f0f0f0" }}>
      {/* main area */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", gap: 10, padding: "10px 60px 0 60px" }}>
        {/* left panel */}
        <div style={panelStyle}>
          <div style={{ color: "#999", fontSize: 10, letterSpacing: 0.5, marginBottom: 12, fontFamily: "Inter, sans-serif" }}>
            experiments
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {data.nodes
              .filter((n) => n.type === "experiment" && (n.experiment_id ?? 0) <= timeSlider)
              .sort((a, b) => (a.experiment_id ?? 0) - (b.experiment_id ?? 0))
              .map((n) => (
                <div
                  key={n.id}
                  onClick={() => setSelection({ kind: "node", data: n })}
                  style={{
                    fontSize: 9,
                    padding: "2px 6px",
                    borderRadius: 3,
                    cursor: "pointer",
                    background: selection?.kind === "node" && selection.data.id === n.id ? "#e0e0e0" : "transparent",
                    color: n.status === "keep" ? "#5b8def" : "#999",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  #{n.experiment_id} {n.change_summary}
                </div>
              ))}
          </div>
        </div>

        {/* graph */}
        <div style={{ flex: 1, position: "relative", background: "#f0f0f0", overflow: "hidden", minWidth: 0, borderRadius: 6 }}>
          <ForceGraph2D
            ref={graphRef}
            graphData={filteredData}
            nodeColor={nodeColor}
            nodeVal={nodeSize}
            nodeLabel={nodeLabel}
            linkColor={linkColor}
            linkWidth={linkWidth}
            linkDirectionalArrowLength={2.5}
            linkDirectionalArrowRelPos={1}
            backgroundColor="#f0f0f0"
            onNodeClick={(node: GraphNode) => setSelection({ kind: "node", data: node })}
            onLinkClick={(link: GraphLink) => setSelection({ kind: "link", data: link })}
            onBackgroundClick={() => setSelection(null)}
            cooldownTime={Infinity}
            d3AlphaDecay={0.005}
            d3AlphaMin={0}
            d3VelocityDecay={0.4}
            linkCurvature={0.1}
            enableNodeDrag={true}
          />
        </div>

        {/* right panel */}
        <div style={panelStyle}>
          <div style={{ color: "#999", fontSize: 10, letterSpacing: 0.5, marginBottom: 12, fontFamily: "Inter, sans-serif" }}>
            {selection ? `${selection.kind} attributes` : "select a node or edge"}
          </div>
          {selection?.kind === "node" && <NodePanel node={selection.data} />}
          {selection?.kind === "link" && <LinkPanel link={selection.data} />}
          {!selection && (
            <div style={{ color: "#bbb", fontSize: 12, marginTop: 20 }}>
              click a node or edge to see its data
            </div>
          )}
        </div>
      </div>

      {/* time slider */}
      <div style={{
        height: 44,
        margin: "10px",
        background: "#e0e0e0",
        border: "1px solid #d4d4d4",
        borderRadius: 6,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 16,
        padding: "0 24px",
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 11, color: "#666" }}>0</span>
        <input
          type="range"
          min={0}
          max={timeMax}
          value={timeSlider}
          onChange={(e) => setTimeSlider(Number(e.target.value))}
          style={{ flex: 1, maxWidth: 800, cursor: "pointer" }}
        />
        <span style={{ fontSize: 11, color: "#666" }}>{timeMax}</span>
        <span style={{ fontSize: 11, color: "#444", marginLeft: 8 }}>
          experiments 0–{timeSlider}
        </span>
      </div>
    </div>
  )
}
