import { useEffect, useRef } from 'react';
import { scaleLinear } from 'd3-scale';
import { axisBottom } from 'd3-axis';
import { select } from 'd3-selection';
import { format } from 'd3-format';

export interface FrequencyAxisProps {
  width: number;
  height?: number;
  minFreqHz: number;
  maxFreqHz: number;
  tickCount?: number;
}

/**
 * Format frequency with SI prefixes
 */
function formatFrequency(hz: number): string {
  if (hz >= 1e9) {
    return `${format('.2f')(hz / 1e9)} GHz`;
  } else if (hz >= 1e6) {
    return `${format('.2f')(hz / 1e6)} MHz`;
  } else if (hz >= 1e3) {
    return `${format('.2f')(hz / 1e3)} kHz`;
  } else {
    return `${format('.0f')(hz)} Hz`;
  }
}

/**
 * Frequency Axis Component
 * D3-based frequency axis with smart formatting
 */
export function FrequencyAxis({
  width,
  height = 40,
  minFreqHz,
  maxFreqHz,
  tickCount = 10,
}: FrequencyAxisProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = select(svgRef.current);
    svg.selectAll('*').remove();

    // Create scale
    const scale = scaleLinear()
      .domain([minFreqHz, maxFreqHz])
      .range([0, width]);

    // Create axis
    const axis = axisBottom(scale)
      .ticks(tickCount)
      .tickFormat((d) => formatFrequency(d as number));

    // Render axis
    const g = svg
      .append('g')
      .attr('transform', `translate(0, 0)`)
      .call(axis);

    // Style axis
    g.selectAll('line')
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.2);

    g.selectAll('path')
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.2);

    g.selectAll('text')
      .attr('fill', 'currentColor')
      .attr('font-size', '11px')
      .attr('font-family', 'IBM Plex Mono, monospace');
  }, [width, minFreqHz, maxFreqHz, tickCount]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="text-muted-foreground"
    />
  );
}
