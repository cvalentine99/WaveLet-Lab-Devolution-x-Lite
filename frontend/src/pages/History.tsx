import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";
import { Link } from "wouter";

export default function History() {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="h-5 w-5" />
              </Button>
            </Link>
            <div>
              <h1 className="text-2xl font-bold">Analysis History</h1>
              <p className="text-sm text-zinc-400">View your past signal recordings</p>
            </div>
          </div>
          {user && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-zinc-400">{user.name}</span>
            </div>
          )}
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Card className="p-8 text-center">
          <h2 className="text-xl font-semibold mb-2">Recording History</h2>
          <p className="text-zinc-400 mb-4">
            View and manage your SigMF recordings from the Live Monitoring page.
          </p>
          <Link href="/live">
            <Button>Go to Live Monitoring</Button>
          </Link>
        </Card>
      </main>
    </div>
  );
}
