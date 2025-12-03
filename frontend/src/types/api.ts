/**
 * API Types - Request/Response types for REST API
 */

/**
 * Paginated list response
 */
export interface PaginatedResponse<T> {
  /** Data items */
  items: T[];

  /** Total count */
  total: number;

  /** Current page */
  page: number;

  /** Page size */
  pageSize: number;

  /** Has next page */
  hasNext: boolean;
}

/**
 * API error response
 */
export interface APIError {
  /** Error message */
  message: string;

  /** Error code */
  code: string;

  /** Additional details */
  details?: Record<string, unknown>;
}
