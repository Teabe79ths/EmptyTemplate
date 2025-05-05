document.addEventListener('DOMContentLoaded', function() {
    // Add animation to flash messages
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(message => {
        // Fade out flash messages after 5 seconds
        setTimeout(() => {
            message.classList.add('fade');
            setTimeout(() => {
                message.remove();
            }, 500);
        }, 5000);
    });

    // Auto resize textarea as user types
    const textareas = document.querySelectorAll('textarea.auto-resize');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Initial resize
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight) + 'px';
    });

    // Character counter for response textarea
    const responseTextarea = document.getElementById('response-textarea');
    const charCounter = document.getElementById('char-counter');
    
    if (responseTextarea && charCounter) {
        responseTextarea.addEventListener('input', function() {
            const remainingChars = 1000 - this.value.length;
            charCounter.textContent = remainingChars;
            
            if (remainingChars < 0) {
                charCounter.classList.add('text-danger');
                document.getElementById('submit-response').disabled = true;
            } else {
                charCounter.classList.remove('text-danger');
                document.getElementById('submit-response').disabled = false;
            }
        });
        
        // Initial count
        const initialCount = 1000 - responseTextarea.value.length;
        charCounter.textContent = initialCount;
    }

    // Confirmation for sensitive actions
    const confirmActions = document.querySelectorAll('.confirm-action');
    confirmActions.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm(this.getAttribute('data-confirm-message') || 'Czy na pewno chcesz to zrobić?')) {
                e.preventDefault();
            }
        });
    });

    // Enable tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
});
