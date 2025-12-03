import { useEffect, useRef } from 'react';
import { scaleLinear } from 'd3-scale';
import { axisLeft } from 'd3-axis';
import { select } from 'd3-selection';
import { format } from 'd3-format';

export interface PowerAxisProps {
  width?: number;
  height: number;
  minPowerDb: number;
  maxPowerDb: number;
  tickCount?: number;
}

/**
 * Power Axis Component
 * D3-based power axis with dB formatting
 */
export function PowerAxis({
  width = 60,
  height,
  minPowerDb,
  maxPowerDb,
  tickCount = 10,
}: PowerAxisProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = select(svgRef.current);
    svg.selectAll('*').remove();

    // Create scale (inverted for top-to-bottom)
    const scale = scaleLinear()
      .domain([maxPowerDb, minPowerDb])
      .range([0, height]);

    // Create axis
    const axis = axisLeft(scale)
      .ticks(tickCount)
      .tickFormat((d) => `${format('.0f')(d as number)} dB`);

    // Render axis
    const g = svg
      .append('g')
      .attr('transform', `translate(${width - 10}, 0)`)
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
  }, [width, height, minPowerDb, maxPowerDb, tickCount]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="text-muted-foreground"
    />
  );
}
