// simple google reviews extractor - run in browser console
// instructions: open reviews page, scroll to load all, paste this code

(() => {
    const reviews = [];
    
    // find all text that looks like reviews by searching for common patterns
    const allElements = document.querySelectorAll('*');
    
    // helper function to check if text looks like a review
    function isValidReview(text) {
        // skip if too short or too long
        if (text.length < 30 || text.length > 2000) return false;
        
        // filter out css/javascript patterns
        const codePatterns = [
            'function(', 'document.', 'querySelector', 'RegExp(',
            '.class{', '@media', 'sentinel{}', 'var(--', 'cursor:',
            'position:', 'background:', 'font-size:', 'padding:',
            'margin:', 'border:', 'display:', 'width:', 'height:',
            'color:', 'opacity:', 'transition:', 'z-index:'
        ];
        
        if (codePatterns.some(pattern => text.includes(pattern))) {
            return false;
        }
        
        // filter out if it's mostly css-like (lots of colons and braces)
        const colonCount = (text.match(/:/g) || []).length;
        const braceCount = (text.match(/[{}]/g) || []).length;
        if (colonCount > 5 && braceCount > 3) return false;
        
        // must contain review-like words (require at least 2)
        const reviewWords = ['food', 'service', 'good', 'great', 'bad', 'nice', 
                            'love', 'hate', 'recommend', 'try', 'order', 'delicious', 
                            'tasty', 'amazing', 'terrible', 'horrible', 'excellent', 
                            'awesome', 'chicken', 'restaurant', 'place', 'visit', 
                            'went', 'came', 'ordered', 'tried', 'staff', 'friendly',
                            'price', 'worth', 'definitely', 'would', 'will', 'burger',
                            'fries', 'sauce', 'meal', 'taste', 'quality', 'experience'];
        
        const foundWords = reviewWords.filter(word => 
            text.toLowerCase().includes(word.toLowerCase())
        );
        
        if (foundWords.length < 2) return false;
        
        // must have some normal sentences (not just technical text)
        const sentences = text.match(/[.!?]\s+/g);
        if (!sentences || sentences.length < 1) {
            // if no sentences, check for normal words vs technical
            const words = text.split(/\s+/);
            const normalWords = words.filter(w => 
                w.length > 2 && /^[a-zA-Z]+$/.test(w)
            );
            if (normalWords.length < 5) return false;
        }
        
        return true;
    }
    
    // helper to check if text is an owner response
    // owner responses contain "(Owner)" in the text, like "Crimson Coward (Owner)4 months ago"
    // or start with "Hi [Name]," which is the owner responding to the reviewer
    function isOwnerResponse(text) {
        // check if text contains owner marker
        if (text.includes('(Owner)')) {
            return true;
        }
        // also check for patterns like "Business Name (Owner)date" at the start
        if (text.match(/^[^(]*\(Owner\)/i)) {
            return true;
        }
        // owner responses often start with "Hi [Name]," where they're addressing the reviewer
        // pattern: "Hi Steph," or "Hi Shawntee," etc.
        if (text.match(/^Hi\s+[A-Z][a-z]+,/i)) {
            return true;
        }
        // check for owner response phrases
        const ownerPhrases = [
            /thanks?\s+for\s+(the|your)\s+(review|feedback|stars?)/i,
            /we\s+(appreciate|love|thank)/i,
            /we['']?ll\s+(take|keep|make)/i,
            /can['']?t\s+wait\s+to\s+serve/i,
            /looking\s+forward\s+to/i,
            /thanks?\s+for\s+the\s+\d+\s*star/i
        ];
        if (ownerPhrases.some(phrase => phrase.test(text))) {
            return true;
        }
        return false;
    }
    
    // helper to extract just the review text, excluding owner responses
    function extractReviewTextOnly(text) {
        // if text contains "(Owner)", split and take only the part before it
        if (text.includes('(Owner)')) {
            const parts = text.split('(Owner)');
            // return the first part (the actual review)
            return parts[0].trim();
        }
        return text;
    }
    
    allElements.forEach(elem => {
        let text = elem.textContent?.trim() || '';
        
        // skip if not a valid review
        if (!isValidReview(text)) return;
        
        // if text contains owner response, try to extract just the review part
        if (text.includes('(Owner)')) {
            // check if there's a review before the owner response
            const reviewText = extractReviewTextOnly(text);
            if (reviewText.length > 30 && isValidReview(reviewText)) {
                text = reviewText; // use just the review part
            } else {
                // if no valid review text before owner response, skip entirely
                return;
            }
        }
        
        // final check - skip if it's purely an owner response
        if (isOwnerResponse(text)) {
            return;
        }
        
        // check if it's already in our list
        if (reviews.some(r => r.text === text || r.text.includes(text) || text.includes(r.text))) {
            return;
        }
        
        // find the review container - look for parent with review-like structure
        let container = elem;
        let reviewContainer = null;
        
        // look for containers that have review structure
        for (let i = 0; i < 5 && container; i++) {
            // check if this container has rating, author, or date elements
            const hasRating = container.querySelector('[aria-label*="star"], [aria-label*="Star"], [role="img"][aria-label*="star"]');
            const hasAuthor = container.querySelector('a[href*="contrib"], a[href*="maps"], div.d4r55, span.X43Kjb');
            const hasDate = container.querySelector('span.rsqaWe, span.leIgZe, [class*="date"], [class*="time"]');
            
            if (hasRating || hasAuthor || hasDate) {
                reviewContainer = container;
                break;
            }
            container = container.parentElement;
        }
        
        // if no container found, use the element's parent
        if (!reviewContainer) {
            reviewContainer = elem.parentElement;
        }
        
        // extract rating from container - try multiple methods
        let rating = null;
        
        // method 1: look for aria-label with star rating
        const ratingSelectors = [
            '[aria-label*="star"]',
            '[aria-label*="Star"]',
            '[aria-label*="Star"]',
            '[role="img"][aria-label*="star"]',
            'span[aria-label*="rating"]',
            '[aria-label*="out of"]',
            '[aria-label*="rating"]'
        ];
        
        for (const selector of ratingSelectors) {
            const ratingElems = reviewContainer.querySelectorAll(selector);
            for (const ratingElem of ratingElems) {
                const ratingText = ratingElem.getAttribute('aria-label') || '';
                // try to extract number from aria-label
                // patterns: "5 stars", "Rated 4 out of 5", "4.0", etc.
                const patterns = [
                    /(\d+)\s*(?:out of|star|stars)/i,
                    /rated\s*(\d+)/i,
                    /(\d+)\s*star/i,
                    /(\d+)/  // just any number
                ];
                
                for (const pattern of patterns) {
                    const match = ratingText.match(pattern);
                    if (match) {
                        const num = parseInt(match[1]);
                        if (num >= 1 && num <= 5) {
                            rating = num;
                            break;
                        }
                    }
                }
                if (rating) break;
            }
            if (rating) break;
        }
        
        // method 2: look for star icons and count them
        if (!rating) {
            const starIcons = reviewContainer.querySelectorAll('[role="img"], svg, [class*="star"]');
            let filledStars = 0;
            for (const star of starIcons) {
                const ariaLabel = star.getAttribute('aria-label') || '';
                const className = (star.className?.toString() || star.getAttribute('class') || '');
                // check if it's a filled star
                if (ariaLabel.includes('star') || className.includes('star')) {
                    // try to extract rating from aria-label
                    const match = ariaLabel.match(/(\d+)/);
                    if (match) {
                        const num = parseInt(match[1]);
                        if (num >= 1 && num <= 5) {
                            rating = num;
                            break;
                        }
                    }
                    // count filled stars (if aria-label says "filled" or similar)
                    if (ariaLabel.includes('filled') || ariaLabel.includes('full') || 
                        !ariaLabel.includes('empty') && !ariaLabel.includes('outline')) {
                        filledStars++;
                    }
                }
            }
            // if we counted stars, use that (assuming 5-star system)
            if (filledStars > 0 && filledStars <= 5 && !rating) {
                rating = filledStars;
            }
        }
        
        // method 3: look for rating in text content
        if (!rating) {
            const containerText = reviewContainer.textContent || '';
            const ratingPatterns = [
                /(\d+)\s*star/i,
                /rated\s*(\d+)/i,
                /(\d+)\s*out\s*of\s*5/i,
                /(\d+)\/5/i,
                /\b([1-5])\b/  // any single digit 1-5 near review text
            ];
            
            for (const pattern of ratingPatterns) {
                const match = containerText.match(pattern);
                if (match) {
                    const num = parseInt(match[1]);
                    if (num >= 1 && num <= 5) {
                        rating = num;
                        break;
                    }
                }
            }
        }
        
        // method 4: look at parent containers more broadly
        if (!rating) {
            let searchContainer = reviewContainer.parentElement;
            for (let i = 0; i < 3 && searchContainer; i++) {
                const ratingElems = searchContainer.querySelectorAll('[aria-label*="star"]');
                for (const elem of ratingElems) {
                    const label = elem.getAttribute('aria-label') || '';
                    const match = label.match(/(\d+)/);
                    if (match) {
                        const num = parseInt(match[1]);
                        if (num >= 1 && num <= 5) {
                            rating = num;
                            break;
                        }
                    }
                }
                if (rating) break;
                searchContainer = searchContainer.parentElement;
            }
        }
        
        // extract author from container
        let author = 'Unknown';
        const authorSelectors = [
            'a[href*="contrib"]',
            'a[href*="maps/contrib"]',
            'div.d4r55',
            'span.X43Kjb',
            'a[href*="maps"]'
        ];
        
        for (const selector of authorSelectors) {
            const authorElem = reviewContainer.querySelector(selector);
            if (authorElem) {
                const authorText = authorElem.textContent.trim();
                // filter out invalid authors
                if (authorText.length > 0 && 
                    authorText.length < 50 && 
                    !authorText.includes('Google') &&
                    !authorText.includes('policy') &&
                    !authorText.match(/^[0-9]+$/) &&
                    !authorText.includes('{') &&
                    !authorText.includes('@') &&
                    !authorText.includes('sentinel')) {
                    author = authorText;
                    break;
                }
            }
        }
        
        // extract date from container - try multiple methods
        let date = 'Unknown';
        
        // method 1: look for specific date elements
        const dateSelectors = [
            'span.rsqaWe',
            'span.leIgZe',
            '[class*="date"]',
            '[class*="time"]',
            'span[data-value]',
            '[class*="rsqaWe"]',
            '[class*="leIgZe"]'
        ];
        
        for (const selector of dateSelectors) {
            const dateElems = reviewContainer.querySelectorAll(selector);
            for (const dateElem of dateElems) {
                const dateText = dateElem.textContent.trim();
                // check if it looks like a date (contains time words)
                if (dateText.match(/\b(ago|month|week|day|year|hour|minute|second|recently|today|yesterday|last|this)\b/i) ||
                    dateText.match(/\d+\s*(month|week|day|year|hour|minute)/i) ||
                    dateText.match(/(a|an)\s+(month|week|day|year|hour)\s+ago/i)) {
                    date = dateText;
                    break;
                }
            }
            if (date !== 'Unknown') break;
        }
        
        // method 2: search all text in container for date patterns
        if (date === 'Unknown') {
            const containerText = reviewContainer.textContent || '';
            const datePatterns = [
                /\d+\s*(month|week|day|year|hour|minute)\s+ago/i,
                /(a|an)\s+(month|week|day|year|hour)\s+ago/i,
                /(\d+)\s*(months?|weeks?|days?|years?|hours?)\s+ago/i,
                /(today|yesterday|recently)/i,
                /(last|this)\s+(month|week|day|year)/i,
                /\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b/i,
                /\d{1,2}\/\d{1,2}\/\d{2,4}/,
                /\d{4}-\d{2}-\d{2}/
            ];
            
            for (const pattern of datePatterns) {
                const match = containerText.match(pattern);
                if (match) {
                    date = match[0].trim();
                    break;
                }
            }
        }
        
        // method 3: look in parent containers
        if (date === 'Unknown') {
            let searchContainer = reviewContainer.parentElement;
            for (let i = 0; i < 4 && searchContainer; i++) {
                const dateElems = searchContainer.querySelectorAll('span, div, time');
                for (const dateElem of dateElems) {
                    const dateText = dateElem.textContent.trim();
                    // check if it's a short text that looks like a date
                    if (dateText.length > 0 && dateText.length < 30 &&
                        (dateText.match(/\b(ago|month|week|day|year|hour|minute|recently|today|yesterday)\b/i) ||
                         dateText.match(/\d+\s*(month|week|day|year|hour)\s*ago/i) ||
                         dateText.match(/(a|an)\s+(month|week|day|year|hour)\s+ago/i))) {
                        date = dateText;
                        break;
                    }
                }
                if (date !== 'Unknown') break;
                searchContainer = searchContainer.parentElement;
            }
        }
        
        // method 4: look for elements with time-related attributes
        if (date === 'Unknown') {
            const timeElems = reviewContainer.querySelectorAll('[datetime], [data-time], time');
            for (const timeElem of timeElems) {
                const timeValue = timeElem.getAttribute('datetime') || 
                                timeElem.getAttribute('data-time') ||
                                timeElem.textContent.trim();
                if (timeValue) {
                    date = timeValue;
                    break;
                }
            }
        }
        
        // filter out reviews with invalid authors
        if (author.includes('Google') || author.includes('policy') || author.length > 50) {
            return;
        }
        
        // final check: make sure text doesn't contain owner response
        // if it does, extract just the review part
        let finalText = text;
        if (text.includes('(Owner)')) {
            // split on "(Owner)" and take only the part before it
            const parts = text.split('(Owner)');
            if (parts[0] && parts[0].trim().length > 30) {
                finalText = parts[0].trim();
            } else {
                // if no valid review text before owner response, skip
                return;
            }
        }
        
        // skip if final text is an owner response (check multiple patterns)
        if (isOwnerResponse(finalText)) {
            return;
        }
        
        // additional check: if text starts with "Hi [Name]," it's likely an owner response
        // unless it's part of a longer review that mentions someone
        if (finalText.match(/^Hi\s+[A-Z][a-z]+,\s*[A-Z]/)) {
            // if it starts with "Hi Name," and then capital letter, it's likely owner response
            // owner responses are usually short and direct
            if (finalText.length < 500) {
                return; // skip short owner responses
            }
        }
        
        reviews.push({
            author: author,
            rating: rating,
            text: finalText,
            date: date
        });
    });
    
    // remove duplicates and filter out bad reviews
    const uniqueReviews = [];
    reviews.forEach(review => {
        // skip if author is invalid
        if (review.author.includes('Google') || review.author.includes('policy')) {
            return;
        }
        
        // skip if it's an owner response (check both full text and if it starts with owner pattern)
        if (isOwnerResponse(review.text)) {
            return;
        }
        
        // also check if text contains owner response - if so, try to extract just review part
        if (review.text.includes('(Owner)')) {
            const reviewOnly = extractReviewTextOnly(review.text);
            if (reviewOnly.length > 30 && isValidReview(reviewOnly)) {
                review.text = reviewOnly; // replace with just review text
            } else {
                return; // skip if no valid review text
            }
        }
        
        // check if it starts with "Hi [Name]," pattern (owner response)
        if (review.text.match(/^Hi\s+[A-Z][a-z]+,\s*[A-Z]/) && review.text.length < 500) {
            return; // skip owner responses
        }
        
        // skip if text looks like css/js
        if (review.text.includes('{') && review.text.includes(':')) {
            const cssLike = (review.text.match(/[{}:;]/g) || []).length;
            if (cssLike > 10) return;
        }
        
        // check for duplicates (keep longest version)
        const existing = uniqueReviews.find(r => {
            const text1 = r.text.toLowerCase().trim();
            const text2 = review.text.toLowerCase().trim();
            // check if texts are very similar (80% overlap)
            return text1.includes(text2.substring(0, Math.min(50, text2.length))) ||
                   text2.includes(text1.substring(0, Math.min(50, text1.length)));
        });
        
        if (!existing) {
            uniqueReviews.push(review);
        } else if (review.text.length > existing.text.length) {
            const index = uniqueReviews.indexOf(existing);
            uniqueReviews[index] = review;
        }
    });
    
    console.log(`\n=== Found ${uniqueReviews.length} reviews ===\n`);
    console.log(JSON.stringify(uniqueReviews, null, 2));
    
    // copy to clipboard
    const json = JSON.stringify(uniqueReviews, null, 2);
    navigator.clipboard.writeText(json).then(() => {
        console.log('\n✓ Copied to clipboard!');
    });
    
    return uniqueReviews;
})();
