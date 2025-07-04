import React from "react";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    const { hasError, error } = this.state;

    if (hasError) {
      return (
        <div className="p-6 bg-red-900 text-white rounded-xl shadow-md max-w-xl mx-auto mt-10">
          <h2 className="text-xl font-bold mb-2">🚨 Something went wrong</h2>
          <pre className="whitespace-pre-wrap text-sm text-red-200">
            {error?.toString()}
          </pre>
          <p className="text-xs text-gray-300 mt-2">
            Please try refreshing the page or check your console for more details.
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
