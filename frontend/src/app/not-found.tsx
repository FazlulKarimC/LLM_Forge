import Link from "next/link";
import { FileQuestion, ArrowLeft } from "lucide-react";

export default function NotFound() {
    return (
        <div className="min-h-screen bg-(--bg-page) flex flex-col items-center justify-center p-4">
            <div className="max-w-md w-full text-center space-y-6">
                {/* Icon & Heading */}
                <div className="flex flex-col items-center justify-center gap-4">
                    <div className="p-4 bg-primary/10 rounded-full text-primary">
                        <FileQuestion className="size-12" strokeWidth={1.5} />
                    </div>
                    <h1 className="text-4xl font-serif text-(--text-heading)">
                        404 - Page Not Found
                    </h1>
                </div>

                {/* Description */}
                <p className="text-(--text-body) text-lg leading-relaxed">
                    The page you are looking for doesn't exist or has been moved.
                    Please check the URL or navigate back home.
                </p>

                {/* Actions */}
                <div className="pt-6 flex flex-col sm:flex-row items-center justify-center gap-4">
                    <Link
                        href="/dashboard"
                        className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-medium text-white bg-primary rounded-lg hover:bg-primary/90 transition-colors shadow-sm w-full sm:w-auto"
                    >
                        <ArrowLeft className="size-4" />
                        Back to Dashboard
                    </Link>
                    <Link
                        href="/experiments"
                        className="inline-flex items-center justify-center px-6 py-3 text-sm font-medium text-(--text-heading) bg-(--bg-card) border border-border rounded-lg hover:bg-(--bg-page) transition-colors shadow-sm w-full sm:w-auto"
                    >
                        View Experiments
                    </Link>
                </div>
            </div>
        </div>
    );
}
