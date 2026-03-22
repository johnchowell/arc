import pytest


@pytest.fixture
def sample_html():
    return """
    <html>
    <head>
        <title>Test Page</title>
        <meta name="description" content="A test page for unit testing">
    </head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <article>
            <h1>Test Article</h1>
            <p>This is a test article about machine learning and artificial intelligence.
            Neural networks have revolutionized many fields including computer vision and
            natural language processing. Deep learning models can now understand images
            and text with remarkable accuracy.</p>
            <img src="https://example.com/image.jpg" alt="Test image">
            <p>Another paragraph with more content about semantic search and embeddings.</p>
        </article>
        <footer>Footer content</footer>
    </body>
    </html>
    """
