import { useState } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useBookmarkStore } from '@/stores/bookmarkStore';
import { ArrowLeft, Bookmark, Trash2, Search, Plus, Image as ImageIcon, Download as DownloadIcon } from 'lucide-react';
import { Link } from 'wouter';
import { toast } from 'sonner';
import { BookmarkDialog } from '@/components/bookmarks/BookmarkDialog';
import { formatDistanceToNow } from 'date-fns';
import { exportBookmarksToCSV, exportBookmarksToJSON } from '@/lib/export';
import { downloadSnapshot } from '@/lib/spectrumCapture';
import { Download, FileDown } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

export default function Bookmarks() {
  const { user } = useAuth();
  const { bookmarks, deleteBookmark, searchBookmarks } = useBookmarkStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddDialog, setShowAddDialog] = useState(false);

  const bookmarkList = searchQuery
    ? searchBookmarks(searchQuery)
    : Array.from(bookmarks.values()).sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

  const handleDelete = (id: string, name: string) => {
    if (confirm(`Delete bookmark "${name}"?`)) {
      deleteBookmark(id);
      toast.success('Bookmark deleted');
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="h-5 w-5" />
                </Button>
              </Link>
              <div>
                <h1 className="text-2xl font-bold">Signal Bookmarks</h1>
                <p className="text-sm text-zinc-400">Saved signals for reference and analysis</p>
              </div>
            </div>
            {user && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-zinc-400">{user.name}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="space-y-6">
          {/* Search and Add */}
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-zinc-500" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search bookmarks by name, notes, tags, or modulation..."
                className="pl-10"
              />
            </div>
            <div className="flex gap-2">
              {bookmarkList.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline">
                      <Download className="w-4 h-4 mr-2" />
                      Export
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem onClick={() => exportBookmarksToCSV(bookmarkList)}>
                      <FileDown className="w-4 h-4 mr-2" />
                      Export as CSV
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => exportBookmarksToJSON(bookmarkList)}>
                      <FileDown className="w-4 h-4 mr-2" />
                      Export as JSON
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
              <Button onClick={() => setShowAddDialog(true)}>
                <Plus className="w-4 h-4 mr-2" />
                Add Bookmark
              </Button>
            </div>
          </div>

          {/* Bookmarks List */}
          {bookmarkList.length === 0 ? (
            <Card className="p-12 text-center">
              <Bookmark className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-semibold mb-2">
                {searchQuery ? 'No bookmarks found' : 'No bookmarks yet'}
              </h3>
              <p className="text-zinc-400 mb-4">
                {searchQuery
                  ? 'Try a different search query'
                  : 'Bookmark interesting signals from Live Monitoring'}
              </p>
              {!searchQuery && (
                <div className="flex gap-2 justify-center">
                  <Button onClick={() => setShowAddDialog(true)} variant="outline">
                    <Plus className="w-4 h-4 mr-2" />
                    Add Bookmark
                  </Button>
                  <Link href="/live">
                    <Button>Go to Live Monitoring</Button>
                  </Link>
                </div>
              )}
            </Card>
          ) : (
            <div className="grid gap-4">
              {bookmarkList.map((bookmark) => (
                <Card key={bookmark.id} className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <Bookmark className="w-5 h-5 text-amber-500" />
                        <h3 className="text-lg font-semibold">{bookmark.name}</h3>
                      </div>

                      <div className="grid md:grid-cols-2 gap-4 mb-3">
                        <div>
                          <div className="text-sm text-zinc-400">Frequency</div>
                          <div className="font-mono text-lg text-primary">
                            {(bookmark.centerFreqHz / 1e6).toFixed(3)} MHz
                          </div>
                        </div>

                        <div>
                          <div className="text-sm text-zinc-400">Power</div>
                          <div className="font-mono text-lg">
                            {bookmark.powerDbm.toFixed(1)} dBm
                            {bookmark.snrDb !== undefined && (
                              <span className="text-sm text-zinc-500 ml-2">
                                SNR: {bookmark.snrDb.toFixed(1)} dB
                              </span>
                            )}
                          </div>
                        </div>

                        {bookmark.bandwidthHz && (
                          <div>
                            <div className="text-sm text-zinc-400">Bandwidth</div>
                            <div className="font-mono">
                              {(bookmark.bandwidthHz / 1e3).toFixed(1)} kHz
                            </div>
                          </div>
                        )}

                        {bookmark.modulationType && (
                          <div>
                            <div className="text-sm text-zinc-400">Modulation</div>
                            <div className="font-mono">{bookmark.modulationType}</div>
                          </div>
                        )}
                      </div>

                      {bookmark.spectrumSnapshot && (
                        <div className="mb-3">
                          <div className="text-sm text-zinc-400 mb-2 flex items-center justify-between">
                            <span>Spectrum Snapshot</span>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                                downloadSnapshot(
                                  bookmark.spectrumSnapshot!,
                                  `spectrum-${bookmark.name.replace(/\s+/g, '-')}-${timestamp}.png`
                                );
                              }}
                            >
                              <DownloadIcon className="w-3 h-3 mr-1" />
                              Download
                            </Button>
                          </div>
                          <div className="border border-zinc-800 rounded overflow-hidden">
                            <img
                              src={bookmark.spectrumSnapshot}
                              alt="Spectrum snapshot"
                              className="w-full h-auto"
                            />
                          </div>
                        </div>
                      )}

                      {bookmark.notes && (
                        <div className="mb-3 p-3 bg-zinc-900/50 rounded border border-zinc-800">
                          <div className="text-sm text-zinc-400 mb-1">Notes</div>
                          <div className="text-sm">{bookmark.notes}</div>
                        </div>
                      )}

                      <div className="flex items-center gap-2 flex-wrap">
                        {bookmark.tags.map((tag) => (
                          <Badge key={tag} variant="secondary">
                            {tag}
                          </Badge>
                        ))}
                        <span className="text-xs text-zinc-500 ml-auto">
                          {formatDistanceToNow(bookmark.createdAt, { addSuffix: true })}
                        </span>
                      </div>
                    </div>

                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDelete(bookmark.id, bookmark.name)}
                      className="ml-4"
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </main>

      <BookmarkDialog open={showAddDialog} onOpenChange={setShowAddDialog} />
    </div>
  );
}
