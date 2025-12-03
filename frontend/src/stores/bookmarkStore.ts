import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SignalBookmark {
  id: string;
  name: string;
  notes?: string;
  centerFreqHz: number;
  bandwidthHz?: number;
  powerDbm: number;
  snrDb?: number;
  modulationType?: string;
  detectionId?: string;
  tags: string[];
  spectrumSnapshot?: string; // Base64 encoded image data
  createdAt: Date;
}

interface BookmarkState {
  bookmarks: Map<string, SignalBookmark>;
  addBookmark: (bookmark: Omit<SignalBookmark, 'id' | 'createdAt'>) => void;
  updateBookmark: (id: string, updates: Partial<SignalBookmark>) => void;
  deleteBookmark: (id: string) => void;
  getBookmarksByFrequencyRange: (freqStartHz: number, freqEndHz: number) => SignalBookmark[];
  searchBookmarks: (query: string) => SignalBookmark[];
}

export const useBookmarkStore = create<BookmarkState>()(
  persist(
    (set, get) => ({
      bookmarks: new Map(),

      addBookmark: (bookmark) => {
        const id = `bookmark_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newBookmark: SignalBookmark = {
          ...bookmark,
          id,
          createdAt: new Date(),
        };

        set((state) => {
          const newBookmarks = new Map(state.bookmarks);
          newBookmarks.set(id, newBookmark);
          return { bookmarks: newBookmarks };
        });
      },

      updateBookmark: (id, updates) => {
        set((state) => {
          const bookmark = state.bookmarks.get(id);
          if (!bookmark) return state;

          const newBookmarks = new Map(state.bookmarks);
          newBookmarks.set(id, { ...bookmark, ...updates });
          return { bookmarks: newBookmarks };
        });
      },

      deleteBookmark: (id) => {
        set((state) => {
          const newBookmarks = new Map(state.bookmarks);
          newBookmarks.delete(id);
          return { bookmarks: newBookmarks };
        });
      },

      getBookmarksByFrequencyRange: (freqStartHz, freqEndHz) => {
        return Array.from(get().bookmarks.values()).filter(
          (b) => b.centerFreqHz >= freqStartHz && b.centerFreqHz <= freqEndHz
        );
      },

      searchBookmarks: (query) => {
        const lowerQuery = query.toLowerCase();
        return Array.from(get().bookmarks.values()).filter(
          (b) =>
            b.name.toLowerCase().includes(lowerQuery) ||
            b.notes?.toLowerCase().includes(lowerQuery) ||
            b.tags.some((tag) => tag.toLowerCase().includes(lowerQuery)) ||
            b.modulationType?.toLowerCase().includes(lowerQuery)
        );
      },
    }),
    {
      name: 'rf-forensics-bookmarks',
      storage: {
        getItem: (name) => {
          const str = localStorage.getItem(name);
          if (!str) return null;
          const parsed = JSON.parse(str);
          return {
            state: {
              ...parsed.state,
              bookmarks: new Map(
                (parsed.state.bookmarks || []).map(([id, bookmark]: [string, any]) => [
                  id,
                  { ...bookmark, createdAt: new Date(bookmark.createdAt) },
                ])
              ),
            },
          };
        },
        setItem: (name, value) => {
          const serialized = {
            state: {
              ...value.state,
              bookmarks: Array.from((value.state.bookmarks as Map<string, SignalBookmark>).entries()),
            },
          };
          localStorage.setItem(name, JSON.stringify(serialized));
        },
        removeItem: (name) => localStorage.removeItem(name),
      },
    }
  )
);
