import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import LiveMonitoring from "./pages/LiveMonitoring";
import Analyze from "./pages/Analyze";
import History from "./pages/History";
import Bookmarks from "./pages/Bookmarks";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/live"} component={LiveMonitoring} />
      <Route path={"/analyze"} component={Analyze} />
      <Route path={"/history"} component={History} />
      <Route path={"/bookmarks"} component={Bookmarks} />
      <Route path={"/404"} component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider
        defaultTheme="dark"
        // switchable
      >
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
